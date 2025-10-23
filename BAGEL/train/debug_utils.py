import torch
from collections import Counter

def debug_packed_batch(batch, tokenizer, new_token_ids, logger, batch_name="Batch"):
    """
    Analyzes a packed batch and logs a summarized, human-readable breakdown of its
    structural blocks. It enforces strict type checking, displays full text content,
    adds separators between samples, and explicitly shows non-standard loss labels.

    Args:
        batch (SimpleCustomBatch): The batch object from the dataloader.
        tokenizer: The tokenizer for decoding.
        new_token_ids (dict): A dictionary of special token IDs.
        logger: The logger object for output.
        batch_name (str): A name for the batch being logged.
    """
    log_lines = []
    log_lines.append("\n" + "="*180)
    log_lines.append(f"  DEBUG: Aggregated Analysis for {batch_name} on Rank 0")
    log_lines.append("="*180)

    # --- 1. Basic Information ---
    sequence_length = batch.sequence_length
    log_lines.append("\n[1. Basic Information]")
    log_lines.append(f"  - Total Sequence Length: {sequence_length}")
    log_lines.append(f"  - Contained Samples: {len(batch.sample_lens)}")
    log_lines.append(f"  - Sample Lengths: {batch.sample_lens}")

    # --- 2. Data Structures for Analysis ---
    packed_text_ids_list = batch.packed_text_ids.tolist() # Renamed for clarity
    packed_pos_ids = batch.packed_position_ids.tolist()
    pos_to_text_id = {pos: token_id for pos, token_id in zip(batch.packed_text_indexes.tolist(), packed_text_ids_list)}
    
    vit_token_indexes = set(batch.packed_vit_token_indexes.tolist() if hasattr(batch, 'packed_vit_token_indexes') else [])
    vae_token_indexes = set(batch.packed_vae_token_indexes.tolist() if hasattr(batch, 'packed_vae_token_indexes') else [])
    ce_loss_indexes = set(batch.ce_loss_indexes.tolist() if hasattr(batch, 'ce_loss_indexes') else [])
    mse_loss_indexes = set(batch.mse_loss_indexes.tolist() if hasattr(batch, 'mse_loss_indexes') else [])

    pos_to_timestep = {}
    if hasattr(batch, 'packed_vae_token_indexes') and hasattr(batch, 'packed_timesteps'):
        all_vae_indices = batch.packed_vae_token_indexes.tolist()
        all_timesteps = batch.packed_timesteps.tolist()
        pos_to_timestep = {pos: ts for pos, ts in zip(all_vae_indices, all_timesteps)}
    
    pos_to_ce_label = {}
    if hasattr(batch, 'ce_loss_indexes') and hasattr(batch, 'packed_label_ids'):
        all_ce_indices = batch.ce_loss_indexes.tolist()
        all_ce_labels = batch.packed_label_ids.tolist()
        if len(all_ce_indices) == len(all_ce_labels):
            pos_to_ce_label = {pos: label_id for pos, label_id in zip(all_ce_indices, all_ce_labels)}
        else:
            logger.warning("!!! WARNING: Mismatch between ce_loss_indexes and packed_label_ids lengths !!!")

    # --- 3. Find Structural Blocks ---
    special_token_values = set(new_token_ids.values())
    boundaries = [0]
    for pos, token_id in pos_to_text_id.items():
        if token_id in special_token_values:
            boundaries.append(pos)
            boundaries.append(pos + 1)
    
    boundaries.append(sequence_length)
    boundaries = sorted(list(set(boundaries)))
    
    sample_end_positions = set()
    current_pos_sum = 0
    for length in batch.sample_lens[:-1]:
        current_pos_sum += length
        sample_end_positions.add(current_pos_sum - 1)

    # --- 4. Aggregated Sequence Table ---
    log_lines.append("\n[2. Aggregated Sequence Table]")
    header = f"{'Block':<6} | {'Position Range':<18} | {'Length':<8} | {'Type':<25} | {'RoPE IDs':<15} | {'Loss (CE/MSE)':<15} | {'Content / Notes'}"
    separator_line = "-" * 180
    log_lines.append(separator_line)
    log_lines.append(header)
    log_lines.append(separator_line)

    block_count = 0
    
    # Helper function to generate special label notes
    def get_special_label_note(pos):
        if pos in ce_loss_indexes:
            # Use the safe pos_to_text_id.get() method instead of indexing a list
            next_token_in_sequence = pos_to_text_id.get(pos + 1)
            target_label_id = pos_to_ce_label.get(pos)
            
            if target_label_id is not None and target_label_id in special_token_values:
                label_token_str = tokenizer.decode([target_label_id])
                return f"Predicts '{label_token_str}'"
        return None

    for i in range(len(boundaries) - 1):
        start_pos, end_pos = boundaries[i], boundaries[i+1] - 1
        
        if start_pos > end_pos: continue
        
        block_count += 1
        block_len = end_pos - start_pos + 1
        pos_range_str = f"[{start_pos}, {end_pos}]"

        block_types_present = set()
        for p in range(start_pos, end_pos + 1):
            if p in vae_token_indexes: block_types_present.add("VAE_IMG")
            elif p in vit_token_indexes: block_types_present.add("VIT_IMG")
            elif p in pos_to_text_id: block_types_present.add("TEXT")
        
        type_str = ", ".join(sorted(list(block_types_present)))
        if len(block_types_present) > 1:
            type_str = f"!!! MIXED: {type_str} !!!"

        block_rope_ids = set(packed_pos_ids[start_pos : end_pos + 1])
        rope_str = f"Single: {list(block_rope_ids)[0]}" if len(block_rope_ids) == 1 else f"Range: [{min(block_rope_ids)}, {max(block_rope_ids)}]"
        
        block_ce_count = sum(1 for p in range(start_pos, end_pos + 1) if p in ce_loss_indexes)
        block_mse_count = sum(1 for p in range(start_pos, end_pos + 1) if p in mse_loss_indexes)
        loss_str = f"{block_ce_count} / {block_mse_count}"
        
        notes = ""
        
        if block_len == 1 and pos_to_text_id.get(start_pos) in special_token_values:
            type_str = "SPECIAL_TOKEN"
            notes = f"Token: {tokenizer.decode([pos_to_text_id[start_pos]])}"
            special_note = get_special_label_note(start_pos)
            if special_note:
                notes += f" | {special_note}"

        elif "TEXT" in block_types_present and len(block_types_present) == 1:
            text_ids_in_block = [pos_to_text_id[p] for p in range(start_pos, end_pos + 1) if p in pos_to_text_id]
            full_text = tokenizer.decode(text_ids_in_block)
            notes = f"Content: \"{full_text}\""
            
            special_labels_notes = []
            for p in range(start_pos, end_pos + 1):
                special_note = get_special_label_note(p)
                if special_note:
                    special_labels_notes.append(f"Pos {p} {special_note}")
            if special_labels_notes:
                notes += f" | Special Labels: {'; '.join(special_labels_notes)}"

        elif "VAE_IMG" in block_types_present and len(block_types_present) == 1:
            if block_mse_count > 0:
                timestep_val = pos_to_timestep.get(start_pos)
                if timestep_val is None:
                    notes = "!!! Target with Loss, but NO Timestep Found !!!"
                elif timestep_val == float('-inf'):
                    notes = "Target Image (Clean, t=0)"
                else:
                    notes = f"Target Image (Noisy, t={timestep_val:.3f})"
            else:
                notes = "Context Image (No Loss)"
        
        log_lines.append(f"{block_count:<6} | {pos_range_str:<18} | {block_len:<8} | {type_str:<25} | {rope_str:<15} | {loss_str:<15} | {notes}")
        
        if end_pos in sample_end_positions:
            log_lines.append(separator_line)
    
    log_lines.append(separator_line)
    
    # --- 5. Overall Statistics ---
    log_lines.append("\n[3. Overall Statistics]")
    stats = {
        "Text Tokens": len(pos_to_text_id),
        "ViT Tokens": len(vit_token_indexes),
        "VAE Tokens": len(vae_token_indexes),
        "CE Loss Tokens": len(ce_loss_indexes),
        "MSE Loss Tokens": len(mse_loss_indexes),
    }
    for name, count in stats.items():
        percentage = (count / sequence_length * 100) if sequence_length > 0 else 0
        log_lines.append(f"  - {name:<18}: {count:<5} ({percentage:5.1f}%)")
    log_lines.append("="*180 + "\n")
    
    logger.info("\n".join(log_lines))