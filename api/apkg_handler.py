import os
import shutil
import sqlite3
import zipfile
import tempfile
from pathlib import Path
import logging
import time
import json

logger = logging.getLogger(__name__)

class ApkgHandler:
    """
    Handler for processing Anki deck files (.apkg)
    Provides functionality to extract, modify, and repackage Anki decks
    """
    
    def __init__(self):
        self.temp_dir = None
        self.collection_path = None
        self.media_dir = None
        self.media_map_path = None
        self.media_files = {}
    
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.collection_path = os.path.join(self.temp_dir, "collection.anki2")
        self.media_dir = os.path.join(self.temp_dir, "media")
        self.media_map_path = os.path.join(self.temp_dir, "media")
        os.makedirs(self.media_dir, exist_ok=True)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def extract_apkg(self, apkg_path):
        """Extract the contents of an .apkg file"""
        try:
            with zipfile.ZipFile(apkg_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            
            if not os.path.exists(self.collection_path):
                raise ValueError("Invalid .apkg file: collection.anki2 not found")
            
            # Load media map if it exists
            if os.path.exists(self.media_map_path):
                with open(self.media_map_path, 'r') as f:
                    self.media_files = json.load(f)
                
        except zipfile.BadZipFile:
            raise ValueError("Invalid .apkg file format")
        except Exception as e:
            raise Exception(f"Error extracting .apkg file: {str(e)}")
    
    def add_tag_to_notes(self, new_tag):
        """Add a tag to all notes in the deck"""
        try:
            conn = sqlite3.connect(self.collection_path)
            cursor = conn.cursor()
            
            # Get all notes
            cursor.execute("SELECT id, tags FROM notes")
            notes = cursor.fetchall()
            
            # Update tags for each note
            for note_id, tags in notes:
                # Process existing tags
                tag_list = tags.strip().split() if tags else []
                if new_tag not in tag_list:
                    tag_list.append(new_tag)
                
                # Update the note with new tags
                new_tags = " ".join(tag_list)
                cursor.execute(
                    "UPDATE notes SET tags = ?, mod = ? WHERE id = ?",
                    (new_tags, int(time.time() * 1000), note_id)
                )
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            raise Exception(f"Database error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error adding tags: {str(e)}")
    
    def create_apkg(self, output_path):
        """Create a new .apkg file with the modified content"""
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add collection database
                zf.write(self.collection_path, "collection.anki2")
                
                # Add media files
                if os.path.exists(self.media_dir):
                    for media_file in os.listdir(self.media_dir):
                        media_path = os.path.join(self.media_dir, media_file)
                        zf.write(media_path, f"media/{media_file}")
                
                # Add media map
                if self.media_files:
                    temp_media_map = os.path.join(self.temp_dir, "media")
                    with open(temp_media_map, 'w') as f:
                        json.dump(self.media_files, f)
                    zf.write(temp_media_map, "media")
                    
        except Exception as e:
            raise Exception(f"Error creating .apkg file: {str(e)}")