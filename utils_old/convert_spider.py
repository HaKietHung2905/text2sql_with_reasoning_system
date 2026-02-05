import json

def convert_spider_json_to_txt(input_json_file, output_txt_file):
    """
    Convert Spider dataset JSON format to tab-separated text format.
    
    Args:
        input_json_file: Path to input JSON file
        output_txt_file: Path to output TXT file
    """
    
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        for item in data:
            # Extract SQL query and database ID
            sql_query = item.get('sql', item.get('query', ''))
            db_id = item.get('db_id', '')
            
            # Write in tab-separated format: SQL_QUERY\tDATABASE_NAME
            f.write(f"{sql_query}\t{db_id}\n")
    
    print(f"Converted {len(data)} entries from {input_json_file} to {output_txt_file}")

def convert_spider_json_questions_to_txt(input_json_file, output_txt_file):
    """
    Convert Spider dataset JSON to questions format for LangChain evaluation.
    
    Args:
        input_json_file: Path to input JSON file
        output_txt_file: Path to output questions TXT file (for --questions parameter)
    """
    
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        for item in data:
            # Extract question and database ID
            question = item.get('question', '')
            db_id = item.get('db_id', '')
            
            # Write in tab-separated format: QUESTION\tDATABASE_NAME
            f.write(f"{question}\t{db_id}\n")
    
    print(f"Converted {len(data)} questions from {input_json_file} to {output_txt_file}")

# Example usage
if __name__ == "__main__":
    # Convert your JSON files
    
    # For gold/reference SQL queries (for --gold parameter)
    convert_spider_json_to_txt('../spider_data/json/dev.json', '../questions/gold_queries.txt')
    
    # For questions (for --questions parameter when using --use_langchain)
    convert_spider_json_questions_to_txt('../spider_data/json/dev.json', '../questions/questions.txt')
    
    print("\nConversion complete!")