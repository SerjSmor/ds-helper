

def visualize_report(report_dict: dict):
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    
    # Extract overall accuracy from report dict
    accuracy = report_dict.get('accuracy', 0.0)
    accuracy_pct = f"{accuracy:.1%}"
    
    # Create panel with accuracy
    accuracy_panel = Panel(
        f"Overall Accuracy: {accuracy_pct}",
        title="Model Performance",
        border_style="green"
    )
    
    # Display the panel
    console.print(accuracy_panel)

