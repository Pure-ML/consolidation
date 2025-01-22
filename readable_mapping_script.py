def reorganize_mapping(original_mapping: dict) -> dict:
    """
    Reorganize a mapping from {original: replacement} format to {replacement: [originals]} format.
    
    Args:
        original_mapping: Dict where keys are original values and values are their replacements
        
    Returns:
        Dict where keys are replacement values and values are lists of original values that map to them
    """
    new_mapping = {}
    
    # Iterate through the original mapping
    for original, replacement in original_mapping.items():
        # If this replacement value isn't in our new mapping yet, initialize its list
        if replacement not in new_mapping:
            new_mapping[replacement] = []
        
        # Add the original value to the list of values being replaced
        # Only add if it's not already in the list (avoid duplicates)
        if original not in new_mapping[replacement]:
            new_mapping[replacement].append(original)
            
    return new_mapping

# Example usage:

# Example original mapping
# original = {
#     "Chevy": "Che./vy",
#     "Che./vy": "Che./vy",
#     "VChevrolet": "Che./vy"
# }

# # Reorganize it
# new = reorganize_mapping(original)

# print("Raw new:")
# print(new)

# # Print the result
# for replacement, originals in new.items():
#     print(f"\n{replacement}:")
#     for original in originals:
#         print(f"  - {original}")