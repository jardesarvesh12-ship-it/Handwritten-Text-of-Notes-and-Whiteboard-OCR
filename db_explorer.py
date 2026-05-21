"""
db_explorer.py - Command-line database explorer for HandwriteAI OCR database.

Usage:
  python db_explorer.py --stats
  python db_explorer.py --list
  python db_explorer.py --view <ID>
  python db_explorer.py --search <term>
  python db_explorer.py --delete <ID>
  python db_explorer.py --export <csv|json> [--out path]
"""

import os
import sqlite3
import argparse
import sys
import json
import csv
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "instance" / "ocr_database.db"

def get_connection():
    if not DB_PATH.exists():
        print(f"Error: Database file not found at {DB_PATH}", file=sys.stderr)
        print("Please run the Flask application once to initialize the database.", file=sys.stderr)
        sys.exit(1)
    return sqlite3.connect(DB_PATH)

def format_cell(val, width):
    val_str = str(val) if val is not None else "NULL"
    # Replace newlines
    val_str = val_str.replace("\n", " ").replace("\r", "")
    if len(val_str) > width:
        return val_str[:width-3] + "..."
    return val_str.ljust(width)

def print_table(headers, rows, col_widths):
    # Print separator
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print(sep)
    
    # Print headers
    hdr_str = "|" + "|".join(f" {format_cell(h, col_widths[i])} " for i, h in enumerate(headers)) + "|"
    print(hdr_str)
    print(sep.replace("-", "="))
    
    # Print rows
    if not rows:
        total_w = sum(col_widths) + len(col_widths) * 3 - 1
        print(f"| {'(No records found)'.center(total_w)} |")
    else:
        for row in rows:
            row_str = "|" + "|".join(f" {format_cell(row[i], col_widths[i])} " for i in range(len(row))) + "|"
            print(row_str)
    print(sep)

def show_stats():
    conn = get_connection()
    c = conn.cursor()
    
    print("\n" + "=" * 50)
    print(" === HANDWRITEAI DATABASE STATISTICS ===")
    print("=" * 50)
    
    # Total records
    c.execute("SELECT COUNT(*) FROM ocr_results")
    total = c.fetchone()[0]
    print(f"Total Saved OCR Records : {total}")
    
    if total == 0:
        print("Database is empty.")
        conn.close()
        return

    # Avg Confidence
    c.execute("SELECT AVG(confidence) FROM ocr_results WHERE confidence IS NOT NULL")
    avg_conf = c.fetchone()[0]
    print(f"Average OCR Confidence  : {avg_conf:.2f}%")
    
    # Engines breakdown
    print("\n--- OCR Engines Distribution ---")
    c.execute("SELECT engine_used, COUNT(*), AVG(confidence) FROM ocr_results GROUP BY engine_used")
    for engine, count, avg in c.fetchall():
        engine_name = engine if engine else "unknown"
        print(f" - {engine_name.upper():<12} : {count:<4} records (Avg Conf: {avg:.1f}%)")
        
    # Pipelines breakdown
    print("\n--- Preprocessing Pipelines Distribution ---")
    c.execute("SELECT pipeline, COUNT(*) FROM ocr_results GROUP BY pipeline")
    for pipeline, count in c.fetchall():
        pipe_name = pipeline if pipeline else "unknown"
        print(f" - {pipe_name:<12} : {count:<4} records")
        
    print("=" * 50 + "\n")
    conn.close()

def list_records(limit=20):
    conn = get_connection()
    c = conn.cursor()
    
    # Query latest records
    c.execute("""
        SELECT id, filename, confidence, engine_used, pipeline, created_at, extracted_text 
        FROM ocr_results 
        ORDER BY created_at DESC 
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    
    headers = ["ID", "Filename", "Conf %", "Engine", "Pipeline", "Created At", "Extracted Text (Snippet)"]
    col_widths = [5, 30, 8, 10, 12, 20, 30]
    
    print(f"\n[LIST] Showing the latest {len(rows)} database records:")
    print_table(headers, rows, col_widths)
    conn.close()

def search_records(query):
    conn = get_connection()
    c = conn.cursor()
    
    # SQLite search
    like_query = f"%{query}%"
    c.execute("""
        SELECT id, filename, confidence, engine_used, pipeline, created_at, extracted_text 
        FROM ocr_results 
        WHERE filename LIKE ? OR extracted_text LIKE ? OR engine_used LIKE ? or pipeline LIKE ?
        ORDER BY created_at DESC
    """, (like_query, like_query, like_query, like_query))
    rows = c.fetchall()
    
    headers = ["ID", "Filename", "Conf %", "Engine", "Pipeline", "Created At", "Extracted Text (Snippet)"]
    col_widths = [5, 30, 8, 10, 12, 20, 30]
    
    print(f"\n[SEARCH] Search results for query '{query}' (Found {len(rows)} matching records):")
    print_table(headers, rows, col_widths)
    conn.close()

def view_record(record_id):
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT id, filename, extracted_text, confidence, engine_used, pipeline, created_at FROM ocr_results WHERE id = ?", (record_id,))
    row = c.fetchone()
    
    if not row:
        print(f"[ERROR] Record with ID {record_id} does not exist.", file=sys.stderr)
        conn.close()
        sys.exit(1)
        
    rid, filename, text, conf, engine, pipeline, created = row
    
    print("\n" + "=" * 60)
    print(f" Detail View — Record ID: {rid}")
    print("=" * 60)
    print(f" Filename      : {filename}")
    print(f" Created At    : {created}")
    print(f" OCR Engine    : {engine.upper() if engine else 'UNKNOWN'}")
    print(f" Pipeline      : {pipeline}")
    print(f" Confidence    : {conf}%")
    print("-" * 60)
    print(" EXTRACTED TEXT:")
    print("-" * 60)
    if text:
        # replace any high unicode quotes or apostrophes to prevent unicode errors
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text)
    else:
        print("(No text was extracted)")
    print("=" * 60 + "\n")
    conn.close()

def delete_record(record_id):
    conn = get_connection()
    c = conn.cursor()
    
    # Check if exists
    c.execute("SELECT filename FROM ocr_results WHERE id = ?", (record_id,))
    row = c.fetchone()
    if not row:
        print(f"[ERROR] Record with ID {record_id} does not exist.", file=sys.stderr)
        conn.close()
        sys.exit(1)
        
    c.execute("DELETE FROM ocr_results WHERE id = ?", (record_id,))
    conn.commit()
    print(f"[DELETE] Successfully deleted record ID {record_id} ('{row[0]}').")
    conn.close()

def export_records(format_type, out_path=None):
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT id, filename, extracted_text, confidence, engine_used, pipeline, created_at FROM ocr_results ORDER BY id ASC")
    rows = c.fetchall()
    
    if not rows:
        print("[WARNING] No records to export.")
        conn.close()
        return
        
    default_name = f"ocr_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}"
    target_path = Path(out_path) if out_path else Path(__file__).resolve().parent / default_name
    
    if format_type == "json":
        data = []
        for r in rows:
            data.append({
                "id": r[0],
                "filename": r[1],
                "extracted_text": r[2],
                "confidence": r[3],
                "engine_used": r[4],
                "pipeline": r[5],
                "created_at": r[6]
            })
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
    elif format_type == "csv":
        with open(target_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Filename", "Extracted Text", "Confidence %", "Engine Used", "Pipeline", "Created At"])
            writer.writerows(rows)
            
    print(f"[SUCCESS] Exported {len(rows)} records to: {target_path}")
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="HandwriteAI - CLI SQLite Database Explorer")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stats", action="store_true", help="Display database summary stats")
    group.add_argument("--list", action="store_true", help="List recent database records")
    group.add_argument("--view", type=int, metavar="ID", help="View full text and metadata for a specific record ID")
    group.add_argument("--search", type=str, metavar="TERM", help="Search records by term in filename or text")
    group.add_argument("--delete", type=int, metavar="ID", help="Delete a record by ID")
    group.add_argument("--export", choices=["csv", "json"], help="Export database to CSV or JSON format")
    
    parser.add_argument("--limit", type=int, default=20, help="Limit number of rows printed in lists (default: 20)")
    parser.add_argument("--out", type=str, help="Custom output filepath for export option")
    
    args = parser.parse_args()
    
    if args.stats:
        show_stats()
    elif args.list:
        list_records(limit=args.limit)
    elif args.view is not None:
        view_record(args.view)
    elif args.search is not None:
        search_records(args.search)
    elif args.delete is not None:
        delete_record(args.delete)
    elif args.export:
        export_records(args.export, args.out)

if __name__ == "__main__":
    main()
