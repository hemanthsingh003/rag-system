"""CLI Interface for RAG System"""

import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

import argparse
import sys
import glob
from pathlib import Path
from rag_engine import get_engine, RAGEngine


def cmd_ingest(args):
    """Ingest document(s) - supports multiple files, glob patterns, and directories"""
    engine = get_engine()
    
    # Collect all file paths
    file_paths = []
    
    for path_str in args.files:
        path = Path(path_str)
        
        if not path.exists():
            print(f"Warning: Path not found: {path_str}")
            continue
        
        if path.is_dir():
            # Directory - get all supported files recursively
            supported = {'.pdf', '.txt', '.md', '.docx', '.png', '.jpg', '.jpeg', '.gif', '.heic', '.heif'}
            for file_path in path.rglob('*'):
                if file_path.suffix.lower() in supported:
                    file_paths.append(str(file_path))
            print(f"Found {len([f for f in path.rglob('*') if f.suffix.lower() in supported])} files in directory: {path_str}")
        elif '*' in path_str or '?' in path_str:
            # Glob pattern
            matches = glob.glob(path_str)
            file_paths.extend(matches)
            print(f"Found {len(matches)} files matching pattern: {path_str}")
        else:
            # Single file
            file_paths.append(path_str)
    
    if not file_paths:
        print("Error: No files to ingest")
        return 1
    
    # Check if multiple files or single file
    if len(file_paths) == 1:
        # Single file - use existing ingest method
        try:
            result = engine.ingest(file_paths[0])
            print(f"\n✓ Document ingested successfully!")
            print(f"  Title: {result['title']}")
            print(f"  Sections: {result['sections']}")
            print(f"  Summary: {result['summary'][:100]}...")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1
    else:
        # Multiple files - use batch ingest
        print(f"\nIngesting {len(file_paths)} files...")
        try:
            result = engine.ingest_batch(file_paths)
            print(f"\n✓ Batch ingestion complete!")
            print(f"  Total processed: {result['total_processed']}")
            print(f"  Total errors: {result['total_errors']}")
            
            if result['total_errors'] > 0:
                print(f"\nErrors:")
                for err in result['errors']:
                    print(f"  - {err['path']}: {err['error']}")
            
            # Show summary of processed files
            if result['results']:
                print(f"\nProcessed files:")
                for r in result['results']:
                    print(f"  - {r['title']}: {r['sections']} sections")
            
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1


def cmd_query(args):
    """Query the RAG system"""
    engine = get_engine()
    show_sources = getattr(args, 'sources', False)
    
    try:
        result = engine.query(args.question)
        
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result['answer'])
        print("="*60)
        
        if show_sources and result['sources']:
            print("\nSOURCES:")
            for i, src in enumerate(result['sources'], 1):
                print(f"\n{i}. {src['title']} (score: {src['score']:.3f})")
                print(f"   {src['text']}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_interactive(args):
    """Interactive mode"""
    engine = get_engine()
    
    print("\n" + "="*60)
    print("RAG System - Interactive Mode")
    print("="*60)
    print("Commands:")
    print("  :ingest <file>  - Ingest a document")
    print("  :query <q>      - Ask a question")
    print("  :sources on/off - Toggle source display")
    print("  :quit           - Exit")
    print("="*60 + "\n")
    
    show_sources = False
    
    while True:
        try:
            cmd = input(">>> ").strip()
            
            if not cmd:
                continue
            
            if cmd.lower() in [":quit", ":q", "exit"]:
                break
            
            if cmd.startswith(":ingest "):
                file_path = cmd[8:].strip()
                try:
                    result = engine.ingest(file_path)
                    print(f"\n✓ Ingested: {result['title']}")
                except Exception as e:
                    print(f"Error: {e}")
            
            elif cmd.startswith(":query ") or cmd.startswith("?"):
                question = cmd[7:].strip() if cmd.startswith(":query ") else cmd[1:].strip()
                try:
                    result = engine.query(question)
                    print(f"\n{result['answer']}")
                    
                    if show_sources and result['sources']:
                        print("\nSources:")
                        for i, src in enumerate(result['sources'], 1):
                            print(f"  {i}. {src['title']} ({src['score']:.3f})")
                except Exception as e:
                    print(f"Error: {e}")
            
            elif cmd == ":sources on":
                show_sources = True
                print("Sources enabled")
            
            elif cmd == ":sources off":
                show_sources = False
                print("Sources disabled")
            
            elif cmd == ":help":
                print("Commands:")
                print("  :ingest <file>  - Ingest a document")
                print("  :query <q>      - Ask a question")
                print("  :sources on/off - Toggle source display")
                print("  :quit           - Exit")
            
            else:
                # Treat as query
                try:
                    result = engine.query(cmd)
                    print(f"\n{result['answer']}")
                except Exception as e:
                    print(f"Error: {e}")
        
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    print("\nGoodbye!")
    return 0


def cmd_server(args):
    """Start API server"""
    print("Starting API server...")
    print(f"Run: python api.py")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest document(s)")
    ingest_parser.add_argument("files", nargs="+", help="File(s), directory, or glob pattern to ingest (PDF, TXT, MD, DOCX, PNG, JPG, JPEG, GIF, HEIC, HEIF)")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("-s", "--sources", action="store_true", 
                             help="Show source documents")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Interactive mode")
    subparsers.add_parser("i", help="Interactive mode (shortcut)")
    
    # Server command
    subparsers.add_parser("server", help="Start API server")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        return cmd_ingest(args)
    elif args.command == "query":
        return cmd_query(args)
    elif args.command in ["interactive", "i"]:
        return cmd_interactive(args)
    elif args.command == "i":
        return cmd_interactive(args)
    elif args.command == "server":
        return cmd_server(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
