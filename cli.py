#!/usr/bin/env python3
import sys
import logging
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML

from src.agent import SableyeAgent
from src.config import Config

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Rich console for beautiful output
console = Console()


def print_banner():
    """Print application banner"""
    banner = """
    ╔═════════════════╗
    ║     Sableye     ║
    ╚═════════════════╝
    """
    console.print(banner, style="bold purple")


def print_help():
    """Print help message"""
    help_text = """
    **Available Commands:**
    
    - Type your question naturally (e.g., "What are my recent goals?")
    - `/help` - Show this help message
    - `/recent [days]` - Show recent entries (default: 7 days)
    - `/stats` - Show vault statistics
    - `/memory` - Show conversation history
    - `/clear` - Clear the screen
    - `/reset` - Reset conversation memory
    - `/exit` or `/quit` - Exit the application
    
    **Example Queries:**
    - What are my goals this month?
    - How has my mood been lately?
    - Am I mentally peaceful recently?
    - What am I anxious about?
    - Show me entries about [topic]
    
    **Tip:** The agent remembers your conversation, so you can ask follow-up 
    questions without repeating context!
    """
    console.print(Panel(Markdown(help_text), title="Help", border_style="yellow"))


def print_stats(agent: SableyeAgent):
    """Print vault statistics"""
    try:
        if not agent.reader:
            console.print("[yellow]No notes loaded yet[/yellow]")
            return
        
        all_docs = agent.reader.read_all_notes()
        recent_docs = agent.reader.read_recent_notes(30)
        
        stats = f"""
        **Vault Statistics:**
        
        - Total Notes: {len(all_docs)}
        - Recent Notes (30 days): {len(recent_docs)}
        - Vault Path: {agent.config.vault.path}
        - Model: {agent.config.model.type} ({agent.config.model.name})
        """
        
        console.print(Panel(Markdown(stats), title="Statistics", border_style="green"))
    
    except Exception as e:
        console.print(f"[red]Error getting stats: {e}[/red]")


@click.group(invoke_without_command=True)
@click.option('--config', '-c', default=None, help='Path to config file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config: str, verbose: bool):
    if ctx.invoked_subcommand is None:
        # Default to chat mode
        ctx.invoke(chat, config=config, verbose=verbose)


@cli.command()
@click.option('--config', '-c', default=None, help='Path to config file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--days', '-d', type=int, help='Number of days to load')
def chat(config: str, verbose: bool, days: Optional[int]):
    """Start interactive chat session"""
    
    print_banner()
    
    try:
        # Load configuration
        console.print("[cyan]Loading configuration...[/cyan]")
        cfg = Config(config)
        
        if verbose:
            cfg.agent.verbose = True
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Validate configuration
        cfg.validate()
        
        # Initialize agent
        console.print("[cyan]Initializing agent...[/cyan]")
        agent = SableyeAgent(cfg)
        
        # Load notes
        with console.status("[cyan]Loading notes from Obsidian vault...[/cyan]"):
            agent.load_notes(days=days)
        
        # Initialize agent
        with console.status("[cyan]Initializing AI agent...[/cyan]"):
            agent.initialize()
        
        console.print("[green]✓ Agent ready![/green]\n")
        
        # Print help
        print_help()
        
        # Chat loop
        console.print("\n[bold]Start chatting![/bold] (Type /help for commands, /exit to quit)\n")
        
        # Create prompt session with history
        history_file = Path.home() / ".sableye_history"
        session = PromptSession(history=FileHistory(str(history_file)))
        
        while True:
            try:
                # Get user input with full readline support
                user_input = session.prompt(HTML("\n<b><ansiblue>You:</ansiblue></b> ")).strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower().split()[0]
                    
                    if command in ['/exit', '/quit', '/q']:
                        console.print("\n[purple]Goodbye! Take care of yourself 💜[/purple]")
                        break
                    
                    elif command == '/help':
                        print_help()
                        continue
                    
                    elif command == '/stats':
                        print_stats(agent)
                        continue
                    
                    elif command == '/clear':
                        console.clear()
                        print_banner()
                        continue
                    
                    elif command == '/reset':
                        agent.clear_memory()
                        console.print("[green]✓ Conversation memory cleared[/green]")
                        continue
                    
                    elif command == '/memory':
                        memory_summary = agent.get_memory_summary()
                        console.print(Panel(
                            memory_summary,
                            title="Conversation History",
                            border_style="cyan"
                        ))
                        continue
                    
                    elif command == '/recent':
                        try:
                            parts = user_input.split()
                            days_arg = int(parts[1]) if len(parts) > 1 else 7
                            user_input = f"Show me my journal entries from the last {days_arg} days"
                        except (ValueError, IndexError):
                            console.print("[red]Invalid format. Use: /recent [days][/red]")
                            continue
                    
                    else:
                        console.print(f"[red]Unknown command: {command}[/red]")
                        continue
                
                # Get agent response
                with console.status("[cyan]Thinking...[/cyan]"):
                    response = agent.chat(user_input)
                
                # Print response
                console.print("\n[bold blue]Sableye[/bold blue]:")
                console.print(Panel(Markdown(response), border_style="green"))
            
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Interrupted. Type /exit to quit.[/yellow]")
                continue
            
            except Exception as e:
                logger.error(f"Error in chat loop: {e}", exc_info=True)
                console.print(f"\n[red]Error: {e}[/red]")
                console.print("[yellow]Please try again or type /exit to quit.[/yellow]")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        console.print(f"\n[red]Fatal error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', default=None, help='Path to config file')
def validate(config: str):
    """Validate configuration"""
    try:
        console.print("[cyan]Validating configuration...[/cyan]")
        cfg = Config(config)
        cfg.validate()
        console.print("[green]✓ Configuration is valid![/green]")
        
        # Print configuration summary
        summary = f"""
        **Configuration Summary:**
        
        - Model Type: {cfg.model.type}
        - Model Name: {cfg.model.name}
        - Vault Path: {cfg.vault.path}
        - Load Days: {cfg.vault.load_days or 'All'}
        - Verbose: {cfg.agent.verbose}
        """
        console.print(Panel(Markdown(summary), title="Config", border_style="cyan"))
    
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--config', '-c', default=None, help='Path to config file')
@click.option('--days', '-d', type=int, help='Number of days to load')
def ask(query: str, config: str, days: Optional[int]):
    """Ask a single question and exit"""
    try:
        # Load configuration
        cfg = Config(config)
        cfg.validate()
        
        # Initialize agent
        with console.status("[cyan]Initializing...[/cyan]"):
            agent = SableyeAgent(cfg)
            agent.load_notes(days=days)
            agent.initialize()
        
        # Get response
        with console.status("[cyan]Thinking...[/cyan]"):
            response = agent.chat(query)
        
        # Print response
        console.print(Panel(Markdown(response), title="Response", border_style="green"))
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()