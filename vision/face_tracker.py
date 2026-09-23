"""Compatibility wrapper — person lock now uses YOLOv8n (person class only)."""
from vision.person_tracker import FaceTrack, FaceTracker, PersonTrack, PersonTracker

__all__ = ["FaceTrack", "FaceTracker", "PersonTrack", "PersonTracker"]
