"""Command-line interface for aspect extraction."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from aspect_extraction import __version__
from aspect_extraction.core.factory import create_extractor

console = Console()


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Aspect Extraction CLI - Extract aspects from text using various methods."""
    pass


@cli.command()
@click.argument("text", type=str)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["auto", "rule", "statistical", "transformer"]),
    default="auto",
    help="Extraction method to use",
)
@click.option(
    "--min-confidence",
    type=float,
    default=0.0,
    help="Minimum confidence threshold (0-1)",
)
@click.option(
    "--no-sentiment",
    is_flag=True,
    help="Disable sentiment analysis",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
def extract(
    text: str,
    method: str,
    min_confidence: float,
    no_sentiment: bool,
    output_json: bool,
) -> None:
    """Extract aspects from TEXT.

    Example:
        aspect-extract extract "The camera quality is great but battery life is poor"
    """
    try:
        # Create extractor
        with console.status(f"[bold green]Extracting aspects using {method} method..."):
            extractor = create_extractor(method=method)  # type: ignore
            aspects = extractor.extract(text)

        # Filter by confidence
        if min_confidence > 0:
            aspects = [a for a in aspects if a.confidence >= min_confidence]

        if not aspects:
            console.print("[yellow]No aspects found.[/yellow]")
            return

        if output_json:
            import json

            output = [a.to_dict() for a in aspects]
            print(json.dumps(output, indent=2))
        else:
            # Create rich table
            table = Table(title=f"Extracted Aspects ({len(aspects)} found)")
            table.add_column("Aspect", style="cyan", no_wrap=True)
            table.add_column("Sentiment", style="magenta")
            table.add_column("Confidence", justify="right", style="green")

            for aspect in aspects:
                sentiment_str = (
                    aspect.sentiment.value if aspect.sentiment and not no_sentiment else "-"
                )
                confidence_str = f"{aspect.confidence:.2f}"

                # Color code sentiment
                if aspect.sentiment:
                    if aspect.sentiment.value == "positive":
                        sentiment_str = f"[green]{sentiment_str}[/green]"
                    elif aspect.sentiment.value == "negative":
                        sentiment_str = f"[red]{sentiment_str}[/red]"

                table.add_row(aspect.text, sentiment_str, confidence_str)

            console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--method",
    "-m",
    type=click.Choice(["auto", "rule", "statistical", "transformer"]),
    default="auto",
    help="Extraction method to use",
)
@click.option(
    "--min-confidence",
    type=float,
    default=0.0,
    help="Minimum confidence threshold (0-1)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (JSON format)",
)
def extract_file(
    file: str,
    method: str,
    min_confidence: float,
    output: Optional[str],
) -> None:
    """Extract aspects from a text FILE.

    The file should contain one text per line.

    Example:
        aspect-extract extract-file reviews.txt -m rule -o results.json
    """
    try:
        # Read file
        file_path = Path(file)
        texts = file_path.read_text().strip().split("\n")
        texts = [t.strip() for t in texts if t.strip()]

        if not texts:
            console.print("[yellow]No texts found in file.[/yellow]")
            return

        # Create extractor
        extractor = create_extractor(method=method)  # type: ignore

        # Process texts with progress bar
        all_aspects = []
        with Progress() as progress:
            task = progress.add_task(f"[cyan]Processing {len(texts)} texts...", total=len(texts))

            for text in texts:
                aspects = extractor.extract(text)

                # Filter by confidence
                if min_confidence > 0:
                    aspects = [a for a in aspects if a.confidence >= min_confidence]

                all_aspects.append(aspects)
                progress.update(task, advance=1)

        # Output results
        total_aspects = sum(len(aspects) for aspects in all_aspects)
        console.print(f"[green]Processed {len(texts)} texts, found {total_aspects} aspects[/green]")

        if output:
            import json

            output_path = Path(output)
            output_data = [[a.to_dict() for a in aspects] for aspects in all_aspects]
            output_path.write_text(json.dumps(output_data, indent=2))
            console.print(f"[green]Results saved to {output}[/green]")
        else:
            # Display summary table
            table = Table(title="Extraction Summary")
            table.add_column("Text", style="cyan", max_width=50)
            table.add_column("Aspects", justify="right", style="green")

            for i, (text, aspects) in enumerate(zip(texts, all_aspects), 1):
                text_preview = text[:50] + "..." if len(text) > 50 else text
                table.add_row(text_preview, str(len(aspects)))

                if i >= 10:  # Limit display to first 10
                    remaining = len(texts) - 10
                    if remaining > 0:
                        table.add_row(f"... and {remaining} more", "")
                    break

            console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command()
def info() -> None:
    """Display information about available extraction methods."""
    info_text = f"""
[bold cyan]Aspect Extraction[/bold cyan] v{__version__}

[bold]Available Methods:[/bold]

[cyan]• rule[/cyan]
  Fast, interpretable rule-based extraction using linguistic patterns.
  Best for: Quick analysis, interpretability, low resource requirements.

[cyan]• statistical[/cyan]
  Frequency and collocation-based extraction.
  Best for: Domain-specific text with repeated aspects.

[cyan]• transformer[/cyan]
  State-of-the-art transformer models (BERT, RoBERTa).
  Best for: Maximum accuracy, diverse text, sufficient compute resources.

[cyan]• auto[/cyan]
  Automatically selects the best available method.

[bold]Usage Examples:[/bold]

  aspect-extract extract "The camera is great" -m rule
  aspect-extract extract-file reviews.txt -m transformer -o results.json
  aspect-extract extract "Battery life is poor" --min-confidence 0.7

[bold]More Information:[/bold]
  GitHub: https://github.com/duyetdev/aspect-extraction
  Documentation: See README.md
    """
    console.print(Panel(info_text, title="Aspect Extraction Info", expand=False))


if __name__ == "__main__":
    cli()
