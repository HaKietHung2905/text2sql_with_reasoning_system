"""
Format and display evaluation results.
"""

from typing import Dict, List


def print_formatted_s(row_name: str, values: List, element_format: str):
    """Print formatted row"""
    template = "{:20} " + ' '.join([element_format] * len(values))
    print(template.format(row_name, *values))


def print_scores(scores: Dict, etype: str, include_turn_acc: bool = True):
    """
    Print evaluation scores in formatted table
    
    Args:
        scores: Dictionary containing all scores
        etype: Evaluation type ('all', 'exec', 'match')
        include_turn_acc: Whether to include turn accuracy
    """
    turns = ['turn 1', 'turn 2', 'turn 3', 'turn 4', 'turn > 4']
    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    if include_turn_acc:
        levels.append('joint_all')
    
    partial_types = [
        'select', 'select(no AGG)', 'where', 'where(no OP)',
        'group(no Having)', 'group', 'order', 'and/or', 'IUEN', 'keywords'
    ]
    
    # Print header
    print_formatted_s("", levels, '{:20}')
    counts = [scores[level]['count'] for level in levels]
    print_formatted_s("count", counts, '{:<20d}')
    
    # Execution accuracy
    if etype in ["all", "exec"]:
        print('\n=====================   EXECUTION ACCURACY     =====================')
        exec_scores = [scores[level]['exec'] for level in levels]
        print_formatted_s("execution", exec_scores, '{:<20.3f}')
    
    # Exact matching
    if etype in ["all", "match"]:
        print('\n====================== EXACT MATCHING ACCURACY =====================')
        exact_scores = [scores[level]['exact'] for level in levels]
        print_formatted_s("exact match", exact_scores, '{:<20.3f}')
        
        print('\n---------------------PARTIAL MATCHING ACCURACY----------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['acc'] for level in levels]
            print_formatted_s(type_, this_scores, '{:<20.3f}')
        
        print('---------------------- PARTIAL MATCHING RECALL ----------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['rec'] for level in levels]
            print_formatted_s(type_, this_scores, '{:<20.3f}')
        
        print('---------------------- PARTIAL MATCHING F1 --------------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['f1'] for level in levels]
            print_formatted_s(type_, this_scores, '{:<20.3f}')
    
    # Turn accuracy
    if include_turn_acc:
        print('\n\n')
        print_formatted_s("", turns, '{:20}')
        counts = [scores[turn]['count'] for turn in turns]
        print_formatted_s("count", counts, "{:<20d}")
        
        if etype in ["all", "exec"]:
            print('\n=====================   TURN EXECUTION ACCURACY     =====================')
            exec_scores = [scores[turn]['exec'] for turn in turns]
            print_formatted_s("execution", exec_scores, '{:<20.3f}')
        
        if etype in ["all", "match"]:
            print('\n====================== TURN EXACT MATCHING ACCURACY =====================')
            exact_scores = [scores[turn]['exact'] for turn in turns]
            print_formatted_s("exact match", exact_scores, '{:<20.3f}')