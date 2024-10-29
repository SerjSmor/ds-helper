from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import json

from ui.visualize import visualize_report


@dataclass
class ClassificationRule:
    class_label: str
    description: str
    words_to_include: List[str]
    words_to_exclude: List[str]


class Analyzer:
    def __init__(self, classification_rules: Dict[str, ClassificationRule] = {}):
        load_dotenv()
        self.client = OpenAI()

    def high_level_error_analysis(self, df: pd.DataFrame, y_column: str, predictions_label: str) -> Tuple[str, str]:
        # Extract true labels and predicted labels
        y_true = df[y_column]
        y_pred = df[predictions_label]

        # Generate classification report
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_text = classification_report(y_true, y_pred)

        # Generate confusion matrix
        labels = sorted(df[y_column].unique())
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual_{label}" for label in labels],
            columns=[f"Predicted_{label}" for label in labels],
        )

        # Prepare the summary of errors
        error_summary = {
            "classification_report": report_dict,
            "confusion_matrix": cm_df.to_dict(),
        }

        # Convert the error summary to a JSON string for compactness
        error_summary_json = json.dumps(error_summary, indent=2)
        # print(error_summary_json)

        # Create the prompt for the GPT model
        prompt = f"""
        You are an NLP expert analyzing the performance of a text classification model. Below are the evaluation metrics:

        Classification Report:
        {report_text}

        Confusion Matrix:
        {cm_df.to_string()}

        Based on the above metrics, provide a detailed analysis of the model's performance. 
        Focus on the worst performing class where the model is performing poorly, discuss possible reasons for misclassifications, 
        and suggest potential improvements.
        Output format should be a json object with the following format: '{{"class_label": "...", "analysis": "..."}}'
        """

        # Call the OpenAI GPT model to get the analysis
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }  # Add this line to enforce JSON response
        )

        # Extract the assistant's reply
        analysis = json.loads(completion.choices[0].message.content)
        # print(analysis)
        return analysis["class_label"], analysis["analysis"], report_dict


    def create_adapt_rules(self, error_analysis: str, df: pd.DataFrame, text_column: str, y_column: str, predictions_label: str, rule: ClassificationRule):

        '''
        Create a list of classification rules based on the predictions and true labels.
        '''
        prompt = f"""
        You are an expert in creating classification rules for a text classification model.
        You are given a dataframe with the true labels and the predicted labels.
        You need to create a list of classification rules based on the predictions and true labels.


        Here is the error analysis:
        {error_analysis}

        Here are the text samples, predictions and the true labels:
        {df[[text_column, y_column, predictions_label]].to_string()}

        The current rule: 
        {rule}

            Please provide your response in the following JSON format:
        {{
        "rule": {{
            "class_label": "string",
            "description": "string",
            "words_to_include": ["word1", "word2", ...],
            "words_to_exclude": ["word1", "word2", ...]
        }},
        "reasoning": "Detailed explanation of why these changes were suggested and how they might improve classification"
        }}

        """


                # Call the OpenAI GPT model to get the analysis
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }  # Add this line to enforce JSON response
        )

        # Parse the JSON response
        response = json.loads(completion.choices[0].message.content)
        
        # Create a ClassificationRule object from the response
        updated_rule = ClassificationRule(
            class_label=response["rule"]["class_label"],
            description=response["rule"]["description"],
            words_to_include=response["rule"]["words_to_include"],
            words_to_exclude=response["rule"]["words_to_exclude"]
        )

        return updated_rule, response["reasoning"]

def apply_rules(df: pd.DataFrame, rules: List[ClassificationRule], text_column: str):    
    def apply_rule(text: str) -> str:
        for rule in rules:
            # Check if text matches rule criteria
            includes_match = all(word.lower() in text.lower() for word in rule.words_to_include)
            excludes_match = all(word.lower() not in text.lower() for word in rule.words_to_exclude)
            
            if includes_match and excludes_match:
                return rule.class_label
        return "other"  # default if no rules match

    # Apply rules and create predictions
    df['lexical_prediction'] = df[text_column].apply(apply_rule)
    return df   

def display_menu(accuracy: float = None, focus_class: str = None, class_accuracy: float = None):
    title = "=== Classification Improvement Menu ==="
    if accuracy is not None:
        title += f" (Overall Accuracy: {accuracy:.2%})"
    if focus_class is not None:
        title += f"\nFocusing on class '{focus_class}' (Accuracy: {class_accuracy:.2%})"
    
    print(f"\n{title}")
    print("1. Run error analysis (using LLM)")
    print("2. View current rules")
    print("3. Create new rules based on error analysis")
    print("4. Calculate and visualize accuracy metrics")
    print("5. Apply rules and check accuracy")
    print("6. Exit")
    return input("Select an option (1-6): ")

def calculate_metrics(df: pd.DataFrame, y_column: str, predictions_label: str) -> Tuple[dict, float]:
    # Generate classification report
    report_dict = classification_report(df[y_column], df[predictions_label], output_dict=True)
    overall_accuracy = report_dict['accuracy']
    visualize_report(report_dict)
    return report_dict, overall_accuracy

def main():
    df = pd.read_csv("data/predictions.csv")
    analyzer = Analyzer()
    rules = []
    last_analysis = None
    last_class_label = None
    overall_accuracy = None
    class_accuracy = None
    
    while True:
        choice = display_menu(overall_accuracy, last_class_label, class_accuracy)
        
        if choice == "1":
            class_label, analysis, report_dict = analyzer.high_level_error_analysis(
                df, y_column="y", predictions_label="predictions"
            )
            last_analysis = analysis
            last_class_label = class_label
            # Calculate class-specific accuracy from the report
            class_accuracy = report_dict[last_class_label]['f1-score']
            # Calculate overall accuracy
            overall_accuracy = report_dict['accuracy']
            print(f"\nAnalysis for class '{class_label}':")
            print(analysis)

        elif choice == "2":
            if not rules:
                print("\nNo rules defined yet.")
            else:
                print("\nCurrent Rules:")
                for i, rule in enumerate(rules, 1):
                    print(f"\nRule {i}:")
                    print(f"Class: {rule.class_label}")
                    print(f"Description: {rule.description}")
                    print(f"Include words: {rule.words_to_include}")
                    print(f"Exclude words: {rule.words_to_exclude}")
                    
        elif choice == "3":
            if not last_analysis:
                print("\nPlease run error analysis first (option 1)")
                continue
                
            df_errors = df[
                (df["y"] != df["predictions"]) & 
                ((df["y"] == last_class_label) | (df["predictions"] == last_class_label))
            ]
            empty_rule = ClassificationRule(
                class_label=last_class_label,
                description="",
                words_to_include=[],
                words_to_exclude=[]
            )
            new_rule, reasoning = analyzer.create_adapt_rules(
                last_analysis,
                df_errors,
                text_column="text",
                y_column="y",
                predictions_label="predictions",
                rule=empty_rule
            )
            rules.append(new_rule)
            print("\nNew rule created:", new_rule)
            print(f"Reasoning: {reasoning}")
            
        elif choice == "4":
            report_dict, overall_accuracy = calculate_metrics(df, y_column="y", predictions_label="predictions")
            if last_class_label:
                class_accuracy = report_dict[last_class_label]['f1-score']
            
        elif choice == "5":
            if not rules:
                print("\nNo rules to apply. Please create rules first (option 3)")
                continue
                
            df_with_rules = apply_rules(df.copy(), rules, text_column="text")
            # Calculate and display accuracy with rules
            report_dict, rule_accuracy = calculate_metrics(
                df_with_rules, 
                y_column="y", 
                predictions_label="lexical_prediction"
            )
            print(df_with_rules.head())
            print(f"\nAccuracy with applied rules: {rule_accuracy:.2%}")
            
        elif choice == "6":
            print("\nExiting program...")
            break
            
        else:
            print("\nInvalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
