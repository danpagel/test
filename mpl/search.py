"""
Search Module for MegaPythonLibrary
===================================

This module provides comprehensive search functionality including:
- Metadata search (size, date, type filters)
- Content-based search capabilities
- Search filters and combinators
- Saved search queries
- Search result caching
- Advanced pattern matching
- Quick convenience functions for common searches

Author: GitHub Copilot
Date: July 2025
"""

import os
import re
import json
import fnmatch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum

from .filesystem import MegaNode, fs_tree, get_node_by_path
from .auth import is_logged_in
from .exceptions import RequestError, ValidationError


# ==============================================
# === SEARCH FILTER DEFINITIONS ===
# ==============================================

class SearchOperator(Enum):
    """Search operators for combining filters."""
    AND = "and"
    OR = "or"
    NOT = "not"


class SizeOperator(Enum):
    """Size comparison operators."""
    EQUALS = "="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    BETWEEN = "between"


class DateOperator(Enum):
    """Date comparison operators."""
    EQUALS = "="
    AFTER = ">"
    BEFORE = "<"
    BETWEEN = "between"
    LAST_DAYS = "last_days"
    LAST_WEEKS = "last_weeks"
    LAST_MONTHS = "last_months"


@dataclass
class SearchFilter:
    """Represents a single search filter."""
    field: str  # name, size, type, date_modified, date_created, path
    operator: str
    value: Any
    case_sensitive: bool = False


@dataclass
class SavedSearch:
    """Represents a saved search query."""
    name: str
    description: str
    filters: List[SearchFilter]
    created_date: datetime
    last_used: datetime
    use_count: int = 0


@dataclass
class SearchResult:
    """Enhanced search result with metadata."""
    node: MegaNode
    relevance_score: float = 1.0
    match_reasons: List[str] = None
    
    def __post_init__(self):
        if self.match_reasons is None:
            self.match_reasons = []


# ==============================================
# === ADVANCED SEARCH ENGINE ===
# ==============================================

class AdvancedSearchEngine:
    """Advanced search engine with filtering, caching, and saved queries."""
    
    def __init__(self, cache_enabled: bool = True):
        """Initialize the advanced search engine."""
        self.cache_enabled = cache_enabled
        self._search_cache = {}
        self._saved_searches = {}
        self._load_saved_searches()
        
        # File type mappings
        self.file_types = {
            'document': ['.txt', '.doc', '.docx', '.pdf', '.rtf', '.odt'],
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp'],
            'video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'],
            'archive': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'],
            'code': ['.py', '.js', '.html', '.css', '.cpp', '.java', '.c', '.h', '.php'],
            'spreadsheet': ['.xls', '.xlsx', '.ods', '.csv'],
            'presentation': ['.ppt', '.pptx', '.odp']
        }
    
    def search(self, 
               query: str = "",
               filters: List[SearchFilter] = None,
               path: str = "/",
               max_results: int = 1000,
               include_folders: bool = True,
               include_files: bool = True) -> List[SearchResult]:
        """
        Perform advanced search with multiple filters.
        
        Args:
            query: Basic text query for file names
            filters: List of SearchFilter objects
            path: Path to search in
            max_results: Maximum number of results
            include_folders: Include folders in results
            include_files: Include files in results
            
        Returns:
            List of SearchResult objects
        """
        if not is_logged_in():
            raise RequestError("Not logged in")
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, filters, path, include_folders, include_files)
        
        # Check cache
        if self.cache_enabled and cache_key in self._search_cache:
            cached_results = self._search_cache[cache_key]
            return cached_results[:max_results]
        
        # Get all nodes to search
        nodes = self._get_search_nodes(path)
        
        # Apply filters
        results = []
        for node in nodes:
            # Type filtering
            if not include_folders and node.is_folder():
                continue
            if not include_files and not node.is_folder():
                continue
            
            # Create search result
            result = SearchResult(node=node)
            
            # Apply text query
            if query and not self._matches_text_query(node, query, result):
                continue
            
            # Apply advanced filters
            if filters and not self._apply_filters(node, filters, result):
                continue
            
            results.append(result)
            
            # Limit results
            if len(results) >= max_results:
                break
        
        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        # Cache results
        if self.cache_enabled:
            self._search_cache[cache_key] = results
        
        return results
    
    def search_by_size(self, 
                      operator: SizeOperator, 
                      size: Union[int, tuple],
                      path: str = "/") -> List[SearchResult]:
        """
        Search files by size.
        
        Args:
            operator: Size comparison operator
            size: Size in bytes (or tuple for BETWEEN)
            path: Path to search in
            
        Returns:
            List of matching files
        """
        filter_obj = SearchFilter(
            field="size",
            operator=operator.value,
            value=size
        )
        return self.search(filters=[filter_obj], path=path, include_folders=False)
    
    def search_by_type(self, file_type: str, path: str = "/") -> List[SearchResult]:
        """
        Search files by type.
        
        Args:
            file_type: File type (document, image, video, etc.) or extension
            path: Path to search in
            
        Returns:
            List of matching files
        """
        filter_obj = SearchFilter(
            field="type",
            operator="=",
            value=file_type
        )
        return self.search(filters=[filter_obj], path=path, include_folders=False)
    
    def search_by_date(self, 
                      operator: DateOperator, 
                      date: Union[datetime, int, tuple],
                      path: str = "/") -> List[SearchResult]:
        """
        Search files by modification date.
        
        Args:
            operator: Date comparison operator
            date: Date object, number of days, or tuple for BETWEEN
            path: Path to search in
            
        Returns:
            List of matching files
        """
        filter_obj = SearchFilter(
            field="date_modified",
            operator=operator.value,
            value=date
        )
        return self.search(filters=[filter_obj], path=path)
    
    def search_with_regex(self, pattern: str, path: str = "/") -> List[SearchResult]:
        """
        Search using regular expressions.
        
        Args:
            pattern: Regular expression pattern
            path: Path to search in
            
        Returns:
            List of matching nodes
        """
        filter_obj = SearchFilter(
            field="name",
            operator="regex",
            value=pattern
        )
        return self.search(filters=[filter_obj], path=path)
    
    def save_search(self, name: str, filters: List[SearchFilter], description: str = "") -> bool:
        """
        Save a search query for later use.
        
        Args:
            name: Name for the saved search
            filters: List of filters to save
            description: Optional description
            
        Returns:
            True if saved successfully
        """
        try:
            saved_search = SavedSearch(
                name=name,
                description=description,
                filters=filters,
                created_date=datetime.now(),
                last_used=datetime.now()
            )
            
            self._saved_searches[name] = saved_search
            self._save_searches_to_file()
            return True
        except Exception as e:
            print(f"Error saving search: {e}")
            return False
    
    def load_saved_search(self, name: str) -> Optional[List[SearchFilter]]:
        """
        Load a saved search query.
        
        Args:
            name: Name of the saved search
            
        Returns:
            List of filters if found, None otherwise
        """
        if name in self._saved_searches:
            saved_search = self._saved_searches[name]
            saved_search.last_used = datetime.now()
            saved_search.use_count += 1
            self._save_searches_to_file()
            return saved_search.filters
        return None
    
    def list_saved_searches(self) -> List[SavedSearch]:
        """Get list of all saved searches."""
        return list(self._saved_searches.values())
    
    def delete_saved_search(self, name: str) -> bool:
        """Delete a saved search."""
        if name in self._saved_searches:
            del self._saved_searches[name]
            self._save_searches_to_file()
            return True
        return False
    
    def clear_cache(self):
        """Clear the search cache."""
        self._search_cache.clear()
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return {
            'cache_size': len(self._search_cache),
            'saved_searches': len(self._saved_searches),
            'cache_enabled': self.cache_enabled,
            'file_types_supported': list(self.file_types.keys())
        }
    
    # ==============================================
    # === PRIVATE METHODS ===
    # ==============================================
    
    def _get_search_nodes(self, path: str) -> List[MegaNode]:
        """Get all nodes to search in the given path."""
        if path == "/":
            nodes = list(fs_tree.nodes.values())
        else:
            start_node = get_node_by_path(path)
            if not start_node:
                return []
            nodes = []
            self._collect_nodes_recursive(start_node, nodes)
        return nodes
    
    def _collect_nodes_recursive(self, node: MegaNode, collection: List[MegaNode]):
        """Recursively collect all nodes."""
        collection.append(node)
        if node.is_folder():
            children = fs_tree.get_children(node.handle)
            for child in children:
                self._collect_nodes_recursive(child, collection)
    
    def _matches_text_query(self, node: MegaNode, query: str, result: SearchResult) -> bool:
        """Check if node matches text query."""
        if not query:
            return True
        
        # Support wildcards
        if '*' in query or '?' in query:
            if fnmatch.fnmatch(node.name.lower(), query.lower()):
                result.match_reasons.append(f"Name matches wildcard: {query}")
                result.relevance_score += 0.8
                return True
        
        # Exact match (highest score)
        if query.lower() == node.name.lower():
            result.match_reasons.append(f"Exact name match: {query}")
            result.relevance_score += 1.0
            return True
        
        # Substring match
        if query.lower() in node.name.lower():
            result.match_reasons.append(f"Name contains: {query}")
            result.relevance_score += 0.6
            return True
        
        # Check file extension
        if query.startswith('.') and node.name.lower().endswith(query.lower()):
            result.match_reasons.append(f"Extension match: {query}")
            result.relevance_score += 0.7
            return True
        
        return False
    
    def _apply_filters(self, node: MegaNode, filters: List[SearchFilter], result: SearchResult) -> bool:
        """Apply advanced filters to a node."""
        for filter_obj in filters:
            if not self._apply_single_filter(node, filter_obj, result):
                return False
        return True
    
    def _apply_single_filter(self, node: MegaNode, filter_obj: SearchFilter, result: SearchResult) -> bool:
        """Apply a single filter to a node."""
        field = filter_obj.field
        operator = filter_obj.operator
        value = filter_obj.value
        
        if field == "name":
            return self._filter_by_name(node, operator, value, filter_obj.case_sensitive, result)
        elif field == "size":
            return self._filter_by_size(node, operator, value, result)
        elif field == "type":
            return self._filter_by_type(node, operator, value, result)
        elif field == "date_modified":
            return self._filter_by_date(node, operator, value, result)
        elif field == "path":
            return self._filter_by_path(node, operator, value, filter_obj.case_sensitive, result)
        
        return True
    
    def _filter_by_name(self, node: MegaNode, operator: str, value: str, case_sensitive: bool, result: SearchResult) -> bool:
        """Filter by file/folder name."""
        name = node.name if case_sensitive else node.name.lower()
        search_value = value if case_sensitive else value.lower()
        
        if operator == "=":
            match = name == search_value
        elif operator == "contains":
            match = search_value in name
        elif operator == "startswith":
            match = name.startswith(search_value)
        elif operator == "endswith":
            match = name.endswith(search_value)
        elif operator == "regex":
            flags = 0 if case_sensitive else re.IGNORECASE
            match = bool(re.search(value, node.name, flags))
        else:
            match = False
        
        if match:
            result.match_reasons.append(f"Name {operator}: {value}")
            result.relevance_score += 0.5
        
        return match
    
    def _filter_by_size(self, node: MegaNode, operator: str, value: Union[int, tuple], result: SearchResult) -> bool:
        """Filter by file size."""
        if node.is_folder():
            return True  # Folders don't have meaningful size
        
        file_size = node.size
        
        if operator == "=":
            match = file_size == value
        elif operator == ">":
            match = file_size > value
        elif operator == "<":
            match = file_size < value
        elif operator == ">=":
            match = file_size >= value
        elif operator == "<=":
            match = file_size <= value
        elif operator == "between":
            if isinstance(value, (list, tuple)) and len(value) == 2:
                match = value[0] <= file_size <= value[1]
            else:
                match = False
        else:
            match = False
        
        if match:
            result.match_reasons.append(f"Size {operator}: {value}")
            result.relevance_score += 0.4
        
        return match
    
    def _filter_by_type(self, node: MegaNode, operator: str, value: str, result: SearchResult) -> bool:
        """Filter by file type."""
        if node.is_folder():
            return value.lower() == "folder"
        
        file_ext = Path(node.name).suffix.lower()
        
        # Check if value is a predefined type
        if value.lower() in self.file_types:
            extensions = self.file_types[value.lower()]
            match = file_ext in extensions
        else:
            # Direct extension match
            if not value.startswith('.'):
                value = '.' + value
            match = file_ext == value.lower()
        
        if match:
            result.match_reasons.append(f"Type: {value}")
            result.relevance_score += 0.5
        
        return match
    
    def _filter_by_date(self, node: MegaNode, operator: str, value: Union[datetime, int, tuple], result: SearchResult) -> bool:
        """Filter by modification date."""
        # Note: Mega API might not provide modification dates
        # This is a placeholder for when that data becomes available
        node_date = getattr(node, 'modification_time', None)
        if not node_date:
            return True  # Skip date filtering if no date available
        
        if isinstance(node_date, (int, float)):
            node_date = datetime.fromtimestamp(node_date)
        
        if operator == "=":
            match = node_date.date() == value.date()
        elif operator == ">":
            match = node_date > value
        elif operator == "<":
            match = node_date < value
        elif operator == "between":
            if isinstance(value, (list, tuple)) and len(value) == 2:
                match = value[0] <= node_date <= value[1]
            else:
                match = False
        elif operator == "last_days":
            cutoff = datetime.now() - timedelta(days=value)
            match = node_date >= cutoff
        elif operator == "last_weeks":
            cutoff = datetime.now() - timedelta(weeks=value)
            match = node_date >= cutoff
        elif operator == "last_months":
            cutoff = datetime.now() - timedelta(days=value * 30)
            match = node_date >= cutoff
        else:
            match = False
        
        if match:
            result.match_reasons.append(f"Date {operator}: {value}")
            result.relevance_score += 0.3
        
        return match
    
    def _filter_by_path(self, node: MegaNode, operator: str, value: str, case_sensitive: bool, result: SearchResult) -> bool:
        """Filter by file path."""
        # Get full path for node (this would need filesystem support)
        node_path = getattr(node, 'full_path', node.name)
        if not case_sensitive:
            node_path = node_path.lower()
            value = value.lower()
        
        if operator == "contains":
            match = value in node_path
        elif operator == "startswith":
            match = node_path.startswith(value)
        elif operator == "endswith":
            match = node_path.endswith(value)
        else:
            match = False
        
        if match:
            result.match_reasons.append(f"Path {operator}: {value}")
            result.relevance_score += 0.3
        
        return match
    
    def _generate_cache_key(self, query: str, filters: List[SearchFilter], path: str, 
                           include_folders: bool, include_files: bool) -> str:
        """Generate a cache key for search parameters."""
        filter_str = ""
        if filters:
            filter_data = [asdict(f) for f in filters]
            filter_str = json.dumps(filter_data, sort_keys=True)
        
        return f"{query}|{filter_str}|{path}|{include_folders}|{include_files}"
    
    def _load_saved_searches(self):
        """Load saved searches from file."""
        try:
            search_file = Path.home() / ".mega_saved_searches.json"
            if search_file.exists():
                with open(search_file, 'r') as f:
                    data = json.load(f)
                    for name, search_data in data.items():
                        # Convert filter dictionaries back to SearchFilter objects
                        filters = [SearchFilter(**f) for f in search_data['filters']]
                        search_data['filters'] = filters
                        search_data['created_date'] = datetime.fromisoformat(search_data['created_date'])
                        search_data['last_used'] = datetime.fromisoformat(search_data['last_used'])
                        self._saved_searches[name] = SavedSearch(**search_data)
        except Exception as e:
            print(f"Error loading saved searches: {e}")
    
    def _save_searches_to_file(self):
        """Save searches to file."""
        try:
            search_file = Path.home() / ".mega_saved_searches.json"
            data = {}
            for name, search in self._saved_searches.items():
                search_dict = asdict(search)
                search_dict['created_date'] = search.created_date.isoformat()
                search_dict['last_used'] = search.last_used.isoformat()
                data[name] = search_dict
            
            with open(search_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving searches: {e}")


# ==============================================
# === CONVENIENCE FUNCTIONS ===
# ==============================================

# Global search engine instance
_search_engine = None

def get_search_engine() -> AdvancedSearchEngine:
    """Get the global search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = AdvancedSearchEngine()
    return _search_engine


def advanced_search(query: str = "", **kwargs) -> List[SearchResult]:
    """Convenience function for advanced search."""
    return get_search_engine().search(query, **kwargs)


def search_by_size(size_mb: float, operator: str = ">=", path: str = "/") -> List[SearchResult]:
    """Search files by size in MB."""
    size_bytes = int(size_mb * 1024 * 1024)
    op = SizeOperator(operator) if isinstance(operator, str) else operator
    return get_search_engine().search_by_size(op, size_bytes, path)


def search_by_extension(extension: str, path: str = "/") -> List[SearchResult]:
    """Search files by extension."""
    if not extension.startswith('.'):
        extension = '.' + extension
    filter_obj = SearchFilter(field="name", operator="endswith", value=extension)
    return get_search_engine().search(filters=[filter_obj], path=path, include_folders=False)


def search_images(path: str = "/") -> List[SearchResult]:
    """Search for image files."""
    return get_search_engine().search_by_type("image", path)


def search_documents(path: str = "/") -> List[SearchResult]:
    """Search for document files."""
    return get_search_engine().search_by_type("document", path)


def search_videos(path: str = "/") -> List[SearchResult]:
    """Search for video files."""
    return get_search_engine().search_by_type("video", path)


def search_audio(path: str = "/") -> List[SearchResult]:
    """Search for audio files."""
    return get_search_engine().search_by_type("audio", path)


def search_recent_files(days: int = 7) -> List[SearchResult]:
    """Search for recently modified files."""
    return get_search_engine().search_by_date(DateOperator.LAST_DAYS, days)


def advanced_search(query: str = "",
                   filters: List[SearchFilter] = None,
                   path: str = "/",
                   max_results: int = 1000,
                   include_folders: bool = True,
                   include_files: bool = True) -> List[SearchResult]:
    """
    Perform advanced search with multiple filters.
    
    Args:
        query: Basic text query for file names
        filters: List of SearchFilter objects
        path: Path to search in
        max_results: Maximum number of results
        include_folders: Include folders in results
        include_files: Include files in results
        
    Returns:
        List of SearchResult objects with relevance scores
    """
    if not is_logged_in():
        raise RequestError("Not logged in")
    
    return get_search_engine().search(
        query=query,
        filters=filters,
        path=path,
        max_results=max_results,
        include_folders=include_folders,
        include_files=include_files
    )


def search_by_type(file_type: str, path: str = "/") -> List[SearchResult]:
    """Search files by type."""
    if not is_logged_in():
        raise RequestError("Not logged in")
    
    return get_search_engine().search_by_type(file_type, path)


def search_with_regex(pattern: str, path: str = "/") -> List[SearchResult]:
    """Search using regular expressions."""
    if not is_logged_in():
        raise RequestError("Not logged in")
    
    return get_search_engine().search_with_regex(pattern, path)


def create_search_query() -> 'SearchQueryBuilder':
    """Create a new search query builder for complex searches."""
    return SearchQueryBuilder(get_search_engine())


def save_search(name: str, filters: List[SearchFilter], description: str = "") -> bool:
    """Save a search query for later use."""
    return get_search_engine().save_search(name, filters, description)


def load_saved_search(name: str) -> Optional[List[SearchFilter]]:
    """Load a saved search query."""
    return get_search_engine().load_saved_search(name)


def list_saved_searches() -> List[SavedSearch]:
    """Get list of all saved searches."""
    return get_search_engine().list_saved_searches()


def delete_saved_search(name: str) -> bool:
    """Delete a saved search."""
    return get_search_engine().delete_saved_search(name)


def get_search_statistics() -> Dict[str, Any]:
    """Get search engine statistics."""
    return get_search_engine().get_search_statistics()


# ==============================================
# === SEARCH QUERY BUILDER ===
# ==============================================

class SearchQueryBuilder:
    """Builder pattern for creating complex search queries."""
    
    def __init__(self, search_engine=None):
        self.search_engine = search_engine
        self.filters = []
        self._query = ""
        self._path = "/"
        self._max_results = 1000
        self._include_folders = True
        self._include_files = True
    
    def query(self, text: str) -> 'SearchQueryBuilder':
        """Set text query."""
        self._query = text
        return self
    
    def in_path(self, path: str) -> 'SearchQueryBuilder':
        """Set search path."""
        self._path = path
        return self
    
    def name_contains(self, text: str, case_sensitive: bool = False) -> 'SearchQueryBuilder':
        """Add name contains filter."""
        self.filters.append(SearchFilter("name", "contains", text, case_sensitive))
        return self
    
    def name_matches(self, pattern: str, case_sensitive: bool = False) -> 'SearchQueryBuilder':
        """Add name exact match filter."""
        self.filters.append(SearchFilter("name", "=", pattern, case_sensitive))
        return self
    
    def name_regex(self, pattern: str, case_sensitive: bool = False) -> 'SearchQueryBuilder':
        """Add regex name filter."""
        self.filters.append(SearchFilter("name", "regex", pattern, case_sensitive))
        return self
    
    def size_greater_than(self, size_mb: float) -> 'SearchQueryBuilder':
        """Add size greater than filter."""
        size_bytes = int(size_mb * 1024 * 1024)
        self.filters.append(SearchFilter("size", ">", size_bytes))
        return self
    
    def size_less_than(self, size_mb: float) -> 'SearchQueryBuilder':
        """Add size less than filter."""
        size_bytes = int(size_mb * 1024 * 1024)
        self.filters.append(SearchFilter("size", "<", size_bytes))
        return self
    
    def size_between(self, min_mb: float, max_mb: float) -> 'SearchQueryBuilder':
        """Add size between filter."""
        min_bytes = int(min_mb * 1024 * 1024)
        max_bytes = int(max_mb * 1024 * 1024)
        self.filters.append(SearchFilter("size", "between", (min_bytes, max_bytes)))
        return self
    
    def file_type(self, type_name: str) -> 'SearchQueryBuilder':
        """Add file type filter."""
        self.filters.append(SearchFilter("type", "=", type_name))
        return self
    
    def extension(self, ext: str) -> 'SearchQueryBuilder':
        """Add extension filter."""
        if not ext.startswith('.'):
            ext = '.' + ext
        self.filters.append(SearchFilter("name", "endswith", ext))
        return self
    
    def folders_only(self) -> 'SearchQueryBuilder':
        """Include only folders."""
        self._include_files = False
        return self
    
    def files_only(self) -> 'SearchQueryBuilder':
        """Include only files."""
        self._include_folders = False
        return self
    
    def limit(self, max_results: int) -> 'SearchQueryBuilder':
        """Set maximum results."""
        self._max_results = max_results
        return self
    
    def execute(self) -> List[SearchResult]:
        """Execute the search query."""
        if self.search_engine:
            return self.search_engine.search(
                query=self._query,
                filters=self.filters,
                path=self._path,
                max_results=self._max_results,
                include_folders=self._include_folders,
                include_files=self._include_files
            )
        else:
            return get_search_engine().search(
                query=self._query,
                filters=self.filters,
                path=self._path,
                max_results=self._max_results,
                include_folders=self._include_folders,
                include_files=self._include_files
            )


def search() -> SearchQueryBuilder:
    """Create a new search query builder."""
    return SearchQueryBuilder()


# Enhanced Search Functions with Events

def advanced_search_with_events(query: str = "", 
                               filters: Optional[List[SearchFilter]] = None,
                               **kwargs) -> List[SearchResult]:
    """Enhanced advanced search with event callbacks and logging.
    
    Args:
        query (str, optional): Search query string. Defaults to "".
        filters (List[SearchFilter], optional): List of search filters to apply. Defaults to None.
        **kwargs: Additional arguments including:
            - callback_fn (Optional[Callable]): Function to call with search events
            - log_fn (Optional[Callable]): Function for logging search operations
            - progress_fn (Optional[Callable]): Function for progress updates
    
    Returns:
        List[SearchResult]: List of matching search results.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    try:
        if callback_fn:
            callback_fn('search_started', {
                'query': query,
                'has_filters': filters is not None,
                'filter_count': len(filters) if filters else 0
            })
        
        if log_fn:
            log_fn(f"Starting advanced search with query: '{query}'")
        
        if progress_fn:
            progress_fn(0, "Initializing search...")
        
        # Delegate to the original function
        results = advanced_search(query, **{k: v for k, v in kwargs.items() 
                                          if k not in ['callback_fn', 'log_fn', 'progress_fn']})
        
        if progress_fn:
            progress_fn(100, f"Search completed - found {len(results)} results")
        
        if callback_fn:
            callback_fn('search_completed', {
                'result_count': len(results),
                'query': query,
                'success': True
            })
        
        if log_fn:
            log_fn(f"Advanced search completed: found {len(results)} results")
        
        return results
        
    except Exception as e:
        if callback_fn:
            callback_fn('search_error', {
                'error': str(e),
                'query': query
            })
        if log_fn:
            log_fn(f"Advanced search error: {e}")
        raise


def search_by_size_with_events(size_mb: float, 
                              operator: str = ">=", 
                              path: str = "/",
                              **kwargs) -> List[SearchResult]:
    """Enhanced size-based search with event callbacks and logging.
    
    Args:
        size_mb (float): Size in megabytes to search for
        operator (str): Comparison operator (">=", "<=", "==", etc.)
        path (str): Path to search within
        **kwargs: Event callback functions
    
    Returns:
        List[SearchResult]: List of matching files by size.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    try:
        if callback_fn:
            callback_fn('size_search_started', {
                'size_mb': size_mb,
                'operator': operator,
                'path': path
            })
        
        if log_fn:
            log_fn(f"Starting size search: {operator} {size_mb}MB in path '{path}'")
        
        if progress_fn:
            progress_fn(0, "Searching by file size...")
        
        results = search_by_size(size_mb, operator, path)
        
        if progress_fn:
            progress_fn(100, f"Size search completed - found {len(results)} files")
        
        if callback_fn:
            callback_fn('size_search_completed', {
                'result_count': len(results),
                'size_mb': size_mb,
                'operator': operator,
                'success': True
            })
        
        if log_fn:
            log_fn(f"Size search completed: found {len(results)} files {operator} {size_mb}MB")
        
        return results
        
    except Exception as e:
        if callback_fn:
            callback_fn('size_search_error', {
                'error': str(e),
                'size_mb': size_mb,
                'operator': operator
            })
        if log_fn:
            log_fn(f"Size search error: {e}")
        raise


def search_by_type_with_events(file_type: str, 
                              path: str = "/",
                              **kwargs) -> List[SearchResult]:
    """Enhanced type-based search with event callbacks and logging.
    
    Args:
        file_type (str): File type to search for (e.g., 'image', 'document', 'video')
        path (str): Path to search within
        **kwargs: Event callback functions
    
    Returns:
        List[SearchResult]: List of matching files by type.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    try:
        if callback_fn:
            callback_fn('type_search_started', {
                'file_type': file_type,
                'path': path
            })
        
        if log_fn:
            log_fn(f"Starting type search for '{file_type}' in path '{path}'")
        
        if progress_fn:
            progress_fn(0, f"Searching for {file_type} files...")
        
        results = search_by_type(file_type, path)
        
        if progress_fn:
            progress_fn(100, f"Type search completed - found {len(results)} {file_type} files")
        
        if callback_fn:
            callback_fn('type_search_completed', {
                'result_count': len(results),
                'file_type': file_type,
                'success': True
            })
        
        if log_fn:
            log_fn(f"Type search completed: found {len(results)} {file_type} files")
        
        return results
        
    except Exception as e:
        if callback_fn:
            callback_fn('type_search_error', {
                'error': str(e),
                'file_type': file_type
            })
        if log_fn:
            log_fn(f"Type search error: {e}")
        raise


def search_by_extension_with_events(extension: str, 
                                   path: str = "/",
                                   **kwargs) -> List[SearchResult]:
    """Enhanced extension-based search with event callbacks and logging.
    
    Args:
        extension (str): File extension to search for (with or without dot)
        path (str): Path to search within
        **kwargs: Event callback functions
    
    Returns:
        List[SearchResult]: List of matching files by extension.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    try:
        if callback_fn:
            callback_fn('extension_search_started', {
                'extension': extension,
                'path': path
            })
        
        if log_fn:
            log_fn(f"Starting extension search for '.{extension.lstrip('.')}' in path '{path}'")
        
        if progress_fn:
            progress_fn(0, f"Searching for .{extension.lstrip('.')} files...")
        
        results = search_by_extension(extension, path)
        
        if progress_fn:
            progress_fn(100, f"Extension search completed - found {len(results)} files")
        
        if callback_fn:
            callback_fn('extension_search_completed', {
                'result_count': len(results),
                'extension': extension,
                'success': True
            })
        
        if log_fn:
            log_fn(f"Extension search completed: found {len(results)} .{extension.lstrip('.')} files")
        
        return results
        
    except Exception as e:
        if callback_fn:
            callback_fn('extension_search_error', {
                'error': str(e),
                'extension': extension
            })
        if log_fn:
            log_fn(f"Extension search error: {e}")
        raise


def search_with_regex_with_events(pattern: str, 
                                 path: str = "/",
                                 **kwargs) -> List[SearchResult]:
    """Enhanced regex search with event callbacks and logging.
    
    Args:
        pattern (str): Regular expression pattern to match filenames
        path (str): Path to search within
        **kwargs: Event callback functions
    
    Returns:
        List[SearchResult]: List of matching files by regex pattern.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    try:
        if callback_fn:
            callback_fn('regex_search_started', {
                'pattern': pattern,
                'path': path
            })
        
        if log_fn:
            log_fn(f"Starting regex search with pattern '{pattern}' in path '{path}'")
        
        if progress_fn:
            progress_fn(0, "Searching with regex pattern...")
        
        results = search_with_regex(pattern, path)
        
        if progress_fn:
            progress_fn(100, f"Regex search completed - found {len(results)} matches")
        
        if callback_fn:
            callback_fn('regex_search_completed', {
                'result_count': len(results),
                'pattern': pattern,
                'success': True
            })
        
        if log_fn:
            log_fn(f"Regex search completed: found {len(results)} files matching '{pattern}'")
        
        return results
        
    except Exception as e:
        if callback_fn:
            callback_fn('regex_search_error', {
                'error': str(e),
                'pattern': pattern
            })
        if log_fn:
            log_fn(f"Regex search error: {e}")
        raise


def search_images_with_events(path: str = "/", **kwargs) -> List[SearchResult]:
    """Enhanced image search with event callbacks and logging.
    
    Args:
        path (str): Path to search within
        **kwargs: Event callback functions
    
    Returns:
        List[SearchResult]: List of image files found.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    try:
        if callback_fn:
            callback_fn('image_search_started', {'path': path})
        
        if log_fn:
            log_fn(f"Starting image search in path '{path}'")
        
        if progress_fn:
            progress_fn(0, "Searching for image files...")
        
        results = search_images(path)
        
        if progress_fn:
            progress_fn(100, f"Image search completed - found {len(results)} images")
        
        if callback_fn:
            callback_fn('image_search_completed', {
                'result_count': len(results),
                'success': True
            })
        
        if log_fn:
            log_fn(f"Image search completed: found {len(results)} image files")
        
        return results
        
    except Exception as e:
        if callback_fn:
            callback_fn('image_search_error', {'error': str(e)})
        if log_fn:
            log_fn(f"Image search error: {e}")
        raise


def search_documents_with_events(path: str = "/", **kwargs) -> List[SearchResult]:
    """Enhanced document search with event callbacks and logging.
    
    Args:
        path (str): Path to search within
        **kwargs: Event callback functions
    
    Returns:
        List[SearchResult]: List of document files found.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    try:
        if callback_fn:
            callback_fn('document_search_started', {'path': path})
        
        if log_fn:
            log_fn(f"Starting document search in path '{path}'")
        
        if progress_fn:
            progress_fn(0, "Searching for document files...")
        
        results = search_documents(path)
        
        if progress_fn:
            progress_fn(100, f"Document search completed - found {len(results)} documents")
        
        if callback_fn:
            callback_fn('document_search_completed', {
                'result_count': len(results),
                'success': True
            })
        
        if log_fn:
            log_fn(f"Document search completed: found {len(results)} document files")
        
        return results
        
    except Exception as e:
        if callback_fn:
            callback_fn('document_search_error', {'error': str(e)})
        if log_fn:
            log_fn(f"Document search error: {e}")
        raise


def search_videos_with_events(path: str = "/", **kwargs) -> List[SearchResult]:
    """Enhanced video search with event callbacks and logging.
    
    Args:
        path (str): Path to search within
        **kwargs: Event callback functions
    
    Returns:
        List[SearchResult]: List of video files found.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    try:
        if callback_fn:
            callback_fn('video_search_started', {'path': path})
        
        if log_fn:
            log_fn(f"Starting video search in path '{path}'")
        
        if progress_fn:
            progress_fn(0, "Searching for video files...")
        
        results = search_videos(path)
        
        if progress_fn:
            progress_fn(100, f"Video search completed - found {len(results)} videos")
        
        if callback_fn:
            callback_fn('video_search_completed', {
                'result_count': len(results),
                'success': True
            })
        
        if log_fn:
            log_fn(f"Video search completed: found {len(results)} video files")
        
        return results
        
    except Exception as e:
        if callback_fn:
            callback_fn('video_search_error', {'error': str(e)})
        if log_fn:
            log_fn(f"Video search error: {e}")
        raise


def search_audio_with_events(path: str = "/", **kwargs) -> List[SearchResult]:
    """Enhanced audio search with event callbacks and logging.
    
    Args:
        path (str): Path to search within
        **kwargs: Event callback functions
    
    Returns:
        List[SearchResult]: List of audio files found.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    progress_fn = kwargs.get('progress_fn')
    
    try:
        if callback_fn:
            callback_fn('audio_search_started', {'path': path})
        
        if log_fn:
            log_fn(f"Starting audio search in path '{path}'")
        
        if progress_fn:
            progress_fn(0, "Searching for audio files...")
        
        results = search_audio(path)
        
        if progress_fn:
            progress_fn(100, f"Audio search completed - found {len(results)} audio files")
        
        if callback_fn:
            callback_fn('audio_search_completed', {
                'result_count': len(results),
                'success': True
            })
        
        if log_fn:
            log_fn(f"Audio search completed: found {len(results)} audio files")
        
        return results
        
    except Exception as e:
        if callback_fn:
            callback_fn('audio_search_error', {'error': str(e)})
        if log_fn:
            log_fn(f"Audio search error: {e}")
        raise


def create_search_query_with_events(**kwargs) -> SearchQueryBuilder:
    """Enhanced search query builder creation with event callbacks and logging.
    
    Args:
        **kwargs: Event callback functions
    
    Returns:
        SearchQueryBuilder: New search query builder instance.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    try:
        if callback_fn:
            callback_fn('query_builder_created', {})
        
        if log_fn:
            log_fn("Creating new search query builder")
        
        builder = create_search_query()
        
        if callback_fn:
            callback_fn('query_builder_ready', {'success': True})
        
        return builder
        
    except Exception as e:
        if callback_fn:
            callback_fn('query_builder_error', {'error': str(e)})
        if log_fn:
            log_fn(f"Search query builder creation error: {e}")
        raise


def save_search_with_events(name: str, 
                          filters: List[SearchFilter], 
                          description: str = "",
                          **kwargs) -> bool:
    """Enhanced search saving with event callbacks and logging.
    
    Args:
        name (str): Name for the saved search
        filters (List[SearchFilter]): Search filters to save
        description (str): Optional description
        **kwargs: Event callback functions
    
    Returns:
        bool: True if search was saved successfully.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    try:
        if callback_fn:
            callback_fn('save_search_started', {
                'name': name,
                'filter_count': len(filters),
                'has_description': bool(description)
            })
        
        if log_fn:
            log_fn(f"Saving search '{name}' with {len(filters)} filters")
        
        success = save_search(name, filters, description)
        
        if callback_fn:
            callback_fn('save_search_completed', {
                'name': name,
                'success': success
            })
        
        if log_fn:
            log_fn(f"Search '{name}' saved: {success}")
        
        return success
        
    except Exception as e:
        if callback_fn:
            callback_fn('save_search_error', {
                'error': str(e),
                'name': name
            })
        if log_fn:
            log_fn(f"Save search error: {e}")
        raise


def load_saved_search_with_events(name: str, **kwargs) -> Optional[List[SearchFilter]]:
    """Enhanced saved search loading with event callbacks and logging.
    
    Args:
        name (str): Name of the saved search to load
        **kwargs: Event callback functions
    
    Returns:
        Optional[List[SearchFilter]]: Search filters if found, None otherwise.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    try:
        if callback_fn:
            callback_fn('load_search_started', {'name': name})
        
        if log_fn:
            log_fn(f"Loading saved search '{name}'")
        
        filters = load_saved_search(name)
        
        if callback_fn:
            callback_fn('load_search_completed', {
                'name': name,
                'found': filters is not None,
                'filter_count': len(filters) if filters else 0
            })
        
        if log_fn:
            log_fn(f"Saved search '{name}' loaded: {'found' if filters else 'not found'}")
        
        return filters
        
    except Exception as e:
        if callback_fn:
            callback_fn('load_search_error', {
                'error': str(e),
                'name': name
            })
        if log_fn:
            log_fn(f"Load saved search error: {e}")
        raise


def list_saved_searches_with_events(**kwargs) -> List[SavedSearch]:
    """Enhanced saved search listing with event callbacks and logging.
    
    Args:
        **kwargs: Event callback functions
    
    Returns:
        List[SavedSearch]: List of all saved searches.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    try:
        if callback_fn:
            callback_fn('list_searches_started', {})
        
        if log_fn:
            log_fn("Listing all saved searches")
        
        searches = list_saved_searches()
        
        if callback_fn:
            callback_fn('list_searches_completed', {
                'count': len(searches),
                'success': True
            })
        
        if log_fn:
            log_fn(f"Listed {len(searches)} saved searches")
        
        return searches
        
    except Exception as e:
        if callback_fn:
            callback_fn('list_searches_error', {'error': str(e)})
        if log_fn:
            log_fn(f"List saved searches error: {e}")
        raise


def delete_saved_search_with_events(name: str, **kwargs) -> bool:
    """Enhanced saved search deletion with event callbacks and logging.
    
    Args:
        name (str): Name of the saved search to delete
        **kwargs: Event callback functions
    
    Returns:
        bool: True if search was deleted successfully.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    try:
        if callback_fn:
            callback_fn('delete_search_started', {'name': name})
        
        if log_fn:
            log_fn(f"Deleting saved search '{name}'")
        
        success = delete_saved_search(name)
        
        if callback_fn:
            callback_fn('delete_search_completed', {
                'name': name,
                'success': success
            })
        
        if log_fn:
            log_fn(f"Saved search '{name}' deleted: {success}")
        
        return success
        
    except Exception as e:
        if callback_fn:
            callback_fn('delete_search_error', {
                'error': str(e),
                'name': name
            })
        if log_fn:
            log_fn(f"Delete saved search error: {e}")
        raise


def get_search_statistics_with_events(**kwargs) -> Dict[str, Any]:
    """Enhanced search statistics with event callbacks and logging.
    
    Args:
        **kwargs: Event callback functions
    
    Returns:
        Dict[str, Any]: Search statistics and metrics.
    """
    callback_fn = kwargs.get('callback_fn')
    log_fn = kwargs.get('log_fn')
    
    try:
        if callback_fn:
            callback_fn('search_stats_started', {})
        
        if log_fn:
            log_fn("Gathering search statistics")
        
        stats = get_search_statistics()
        
        if callback_fn:
            callback_fn('search_stats_completed', {
                'stats_available': len(stats) > 0,
                'success': True
            })
        
        if log_fn:
            log_fn(f"Search statistics gathered: {len(stats)} metrics available")
        
        return stats
        
    except Exception as e:
        if callback_fn:
            callback_fn('search_stats_error', {'error': str(e)})
        if log_fn:
            log_fn(f"Search statistics error: {e}")
        raise


# ==============================================
# === CLIENT METHOD INJECTION ===
# ==============================================

def add_search_methods_with_events(client_class):
    """Add search methods with event support to the MPLClient class."""
    
    def advanced_search_method(self, query: str = "", **kwargs):
        """Perform advanced search with enhanced event callbacks."""
        return advanced_search_with_events(query, 
                                         callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def search_by_size_method(self, size_mb: float, operator: str = ">=", path: str = "/", **kwargs):
        """Search files by size with enhanced event callbacks."""
        return search_by_size_with_events(size_mb, operator, path, 
                                         callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def search_by_type_method(self, file_type: str, path: str = "/", **kwargs):
        """Search files by type with enhanced event callbacks."""
        return search_by_type_with_events(file_type, path, 
                                        callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def search_by_extension_method(self, extension: str, path: str = "/", **kwargs):
        """Search files by extension with enhanced event callbacks."""
        return search_by_extension_with_events(extension, path, 
                                             callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def search_with_regex_method(self, pattern: str, path: str = "/", **kwargs):
        """Search using regular expressions with enhanced event callbacks."""
        return search_with_regex_with_events(pattern, path, 
                                           callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def search_images_method(self, path: str = "/", **kwargs):
        """Search for image files with enhanced event callbacks."""
        return search_images_with_events(path, 
                                       callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def search_documents_method(self, path: str = "/", **kwargs):
        """Search for document files with enhanced event callbacks."""
        return search_documents_with_events(path, 
                                          callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def search_videos_method(self, path: str = "/", **kwargs):
        """Search for video files with enhanced event callbacks."""
        return search_videos_with_events(path, 
                                       callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def search_audio_method(self, path: str = "/", **kwargs):
        """Search for audio files with enhanced event callbacks."""
        return search_audio_with_events(path, 
                                      callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def create_search_query_method(self, **kwargs):
        """Create a new search query builder for complex searches with enhanced event callbacks."""
        return create_search_query_with_events(
            callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def save_search_method(self, name: str, filters, description: str = "", **kwargs):
        """Save a search query for later use with enhanced event callbacks."""
        return save_search_with_events(name, filters, description, 
                                     callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def load_saved_search_method(self, name: str, **kwargs):
        """Load a saved search query with enhanced event callbacks."""
        return load_saved_search_with_events(name, 
                                           callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def list_saved_searches_method(self, **kwargs):
        """List all saved search queries with enhanced event callbacks."""
        return list_saved_searches_with_events(
            callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def delete_saved_search_method(self, name: str, **kwargs):
        """Delete a saved search query with enhanced event callbacks."""
        return delete_saved_search_with_events(name, 
                                             callback_fn=getattr(self, '_trigger_event', None), **kwargs)
    
    def get_search_statistics_method(self, **kwargs):
        """Get search statistics with enhanced event callbacks."""
        return get_search_statistics_with_events(
            callback_fn=getattr(self, '_trigger_event', None), **kwargs)

    # Add methods to client class
    setattr(client_class, 'advanced_search', advanced_search_method)
    setattr(client_class, 'search_by_size', search_by_size_method)
    setattr(client_class, 'search_by_type', search_by_type_method)
    setattr(client_class, 'search_by_extension', search_by_extension_method)
    setattr(client_class, 'search_with_regex', search_with_regex_method)
    setattr(client_class, 'search_images', search_images_method)
    setattr(client_class, 'search_documents', search_documents_method)
    setattr(client_class, 'search_videos', search_videos_method)
    setattr(client_class, 'search_audio', search_audio_method)
    setattr(client_class, 'create_search_query', create_search_query_method)
    setattr(client_class, 'save_search', save_search_method)
    setattr(client_class, 'load_saved_search', load_saved_search_method)
    setattr(client_class, 'list_saved_searches', list_saved_searches_method)
    setattr(client_class, 'delete_saved_search', delete_saved_search_method)
    setattr(client_class, 'get_search_statistics', get_search_statistics_method)

