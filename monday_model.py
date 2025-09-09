"""
Monday.com API Model
A comprehensive model for interacting with Monday.com API
"""

import requests
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MondayItem:
    """Represents a Monday.com item"""
    id: Optional[str] = None
    name: Optional[str] = None
    column_values: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    state: Optional[str] = None
    board_id: Optional[str] = None
    group_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests"""
        return asdict(self)


@dataclass
class MondayBoard:
    """Represents a Monday.com board"""
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    state: Optional[str] = None
    workspace_id: Optional[str] = None
    owner_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    columns: Optional[List[Dict[str, Any]]] = None
    groups: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests"""
        return asdict(self)


@dataclass
class MondayUser:
    """Represents a Monday.com user"""
    id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    enabled: Optional[bool] = None
    is_admin: Optional[bool] = None
    is_guest: Optional[bool] = None
    is_view_only: Optional[bool] = None
    is_limited: Optional[bool] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests"""
        return asdict(self)


class MondayModel:
    """
    Main model class for Monday.com API interactions
    """
    
    def __init__(self, api_token: str, api_version: str = "2023-10"):
        """
        Initialize Monday.com model
        
        Args:
            api_token: Monday.com API token
            api_version: API version to use
        """
        self.api_token = api_token
        self.api_version = api_version
        self.base_url = "https://api.monday.com/v2"
        self.headers = {
            "Authorization": api_token,
            "Content-Type": "application/json",
            "API-Version": api_version
        }
    
    def _make_request(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GraphQL request to Monday.com API
        
        Args:
            query: GraphQL query string
            variables: Variables for the query
            
        Returns:
            API response as dictionary
        """
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "errors" in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                raise Exception(f"GraphQL errors: {data['errors']}")
            
            return data.get("data", {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise
    
    def get_boards(self, limit: int = 50, page: int = 1) -> List[MondayBoard]:
        """
        Get all boards
        
        Args:
            limit: Number of boards to retrieve
            page: Page number for pagination
            
        Returns:
            List of MondayBoard objects
        """
        query = """
        query GetBoards($limit: Int, $page: Int) {
            boards(limit: $limit, page: $page) {
                id
                name
                description
                state
                workspace {
                    id
                }
                owner {
                    id
                }
                created_at
                updated_at
                columns {
                    id
                    title
                    type
                    settings_str
                }
                groups {
                    id
                    title
                    color
                }
            }
        }
        """
        
        variables = {"limit": limit, "page": page}
        data = self._make_request(query, variables)
        
        boards = []
        for board_data in data.get("boards", []):
            board = MondayBoard(
                id=board_data.get("id"),
                name=board_data.get("name"),
                description=board_data.get("description"),
                state=board_data.get("state"),
                workspace_id=board_data.get("workspace", {}).get("id") if board_data.get("workspace") else None,
                owner_id=board_data.get("owner", {}).get("id") if board_data.get("owner") else None,
                created_at=board_data.get("created_at"),
                updated_at=board_data.get("updated_at"),
                columns=board_data.get("columns", []),
                groups=board_data.get("groups", [])
            )
            boards.append(board)
        
        return boards
    
    def get_board_by_id(self, board_id: str) -> Optional[MondayBoard]:
        """
        Get a specific board by ID
        
        Args:
            board_id: Board ID
            
        Returns:
            MondayBoard object or None if not found
        """
        query = """
        query GetBoard($boardId: ID!) {
            boards(ids: [$boardId]) {
                id
                name
                description
                state
                workspace {
                    id
                }
                owner {
                    id
                }
                created_at
                updated_at
                columns {
                    id
                    title
                    type
                    settings_str
                }
                groups {
                    id
                    title
                    color
                }
            }
        }
        """
        
        variables = {"boardId": board_id}
        data = self._make_request(query, variables)
        
        boards = data.get("boards", [])
        if not boards:
            return None
        
        board_data = boards[0]
        return MondayBoard(
            id=board_data.get("id"),
            name=board_data.get("name"),
            description=board_data.get("description"),
            state=board_data.get("state"),
            workspace_id=board_data.get("workspace", {}).get("id") if board_data.get("workspace") else None,
            owner_id=board_data.get("owner", {}).get("id") if board_data.get("owner") else None,
            created_at=board_data.get("created_at"),
            updated_at=board_data.get("updated_at"),
            columns=board_data.get("columns", []),
            groups=board_data.get("groups", [])
        )
    
    def get_items(self, board_id: str, limit: int = 50, page: int = 1) -> List[MondayItem]:
        """
        Get items from a specific board
        
        Args:
            board_id: Board ID
            limit: Number of items to retrieve
            page: Page number for pagination
            
        Returns:
            List of MondayItem objects
        """
        query = """
        query GetItems($boardId: ID!, $limit: Int, $page: Int) {
            boards(ids: [$boardId]) {
                items_page(limit: $limit, page: $page) {
                    cursor
                    items {
                        id
                        name
                        column_values {
                            id
                            text
                            value
                        }
                        created_at
                        updated_at
                        state
                        group {
                            id
                        }
                    }
                }
            }
        }
        """
        
        variables = {"boardId": board_id, "limit": limit, "page": page}
        data = self._make_request(query, variables)
        
        items = []
        boards = data.get("boards", [])
        if boards:
            items_page = boards[0].get("items_page", {})
            for item_data in items_page.get("items", []):
                item = MondayItem(
                    id=item_data.get("id"),
                    name=item_data.get("name"),
                    column_values=item_data.get("column_values", {}),
                    created_at=item_data.get("created_at"),
                    updated_at=item_data.get("updated_at"),
                    state=item_data.get("state"),
                    board_id=board_id,
                    group_id=item_data.get("group", {}).get("id") if item_data.get("group") else None
                )
                items.append(item)
        
        return items
    
    def create_item(self, board_id: str, item_name: str, 
                   column_values: Optional[Dict[str, Any]] = None,
                   group_id: Optional[str] = None) -> MondayItem:
        """
        Create a new item in a board
        
        Args:
            board_id: Board ID
            item_name: Name of the item
            column_values: Column values for the item
            group_id: Group ID (optional)
            
        Returns:
            Created MondayItem object
        """
        query = """
        mutation CreateItem($boardId: ID!, $itemName: String!, $columnValues: JSON, $groupId: String) {
            create_item(board_id: $boardId, item_name: $itemName, column_values: $columnValues, group_id: $groupId) {
                id
                name
                column_values {
                    id
                    text
                    value
                }
                created_at
                updated_at
                state
                group {
                    id
                }
            }
        }
        """
        
        variables = {
            "boardId": board_id,
            "itemName": item_name,
            "columnValues": json.dumps(column_values) if column_values else None,
            "groupId": group_id
        }
        
        data = self._make_request(query, variables)
        item_data = data.get("create_item", {})
        
        return MondayItem(
            id=item_data.get("id"),
            name=item_data.get("name"),
            column_values=item_data.get("column_values", {}),
            created_at=item_data.get("created_at"),
            updated_at=item_data.get("updated_at"),
            state=item_data.get("state"),
            board_id=board_id,
            group_id=item_data.get("group", {}).get("id") if item_data.get("group") else None
        )
    
    def update_item(self, item_id: str, column_values: Dict[str, Any]) -> MondayItem:
        """
        Update an existing item
        
        Args:
            item_id: Item ID
            column_values: Column values to update
            
        Returns:
            Updated MondayItem object
        """
        query = """
        mutation UpdateItem($itemId: ID!, $columnValues: JSON!) {
            change_column_value(item_id: $itemId, column_id: $columnId, value: $value) {
                id
                name
                column_values {
                    id
                    text
                    value
                }
                updated_at
            }
        }
        """
        
        # For simplicity, this updates one column at a time
        # In practice, you might want to batch multiple column updates
        updated_item = None
        
        for column_id, value in column_values.items():
            variables = {
                "itemId": item_id,
                "columnId": column_id,
                "value": json.dumps(value)
            }
            
            data = self._make_request(query, variables)
            item_data = data.get("change_column_value", {})
            
            if item_data:
                updated_item = MondayItem(
                    id=item_data.get("id"),
                    name=item_data.get("name"),
                    column_values=item_data.get("column_values", {}),
                    updated_at=item_data.get("updated_at")
                )
        
        return updated_item
    
    def delete_item(self, item_id: str) -> bool:
        """
        Delete an item
        
        Args:
            item_id: Item ID
            
        Returns:
            True if successful, False otherwise
        """
        query = """
        mutation DeleteItem($itemId: ID!) {
            delete_item(item_id: $itemId) {
                id
            }
        }
        """
        
        variables = {"itemId": item_id}
        
        try:
            data = self._make_request(query, variables)
            return bool(data.get("delete_item", {}).get("id"))
        except Exception as e:
            logger.error(f"Failed to delete item {item_id}: {e}")
            return False
    
    def get_users(self, limit: int = 50, page: int = 1) -> List[MondayUser]:
        """
        Get all users
        
        Args:
            limit: Number of users to retrieve
            page: Page number for pagination
            
        Returns:
            List of MondayUser objects
        """
        query = """
        query GetUsers($limit: Int, $page: Int) {
            users(limit: $limit, page: $page) {
                id
                name
                email
                enabled
                is_admin
                is_guest
                is_view_only
                is_limited
                created_at
                updated_at
            }
        }
        """
        
        variables = {"limit": limit, "page": page}
        data = self._make_request(query, variables)
        
        users = []
        for user_data in data.get("users", []):
            user = MondayUser(
                id=user_data.get("id"),
                name=user_data.get("name"),
                email=user_data.get("email"),
                enabled=user_data.get("enabled"),
                is_admin=user_data.get("is_admin"),
                is_guest=user_data.get("is_guest"),
                is_view_only=user_data.get("is_view_only"),
                is_limited=user_data.get("is_limited"),
                created_at=user_data.get("created_at"),
                updated_at=user_data.get("updated_at")
            )
            users.append(user)
        
        return users


# Example usage
if __name__ == "__main__":
    # Initialize the model with your API token
    # api_token = "your_monday_api_token_here"
    # monday = MondayModel(api_token)
    
    # Example operations:
    # boards = monday.get_boards()
    # board = monday.get_board_by_id("board_id_here")
    # items = monday.get_items("board_id_here")
    # new_item = monday.create_item("board_id_here", "New Item", {"status": "Working on it"})
    
    print("Monday.com Model initialized. Please provide your API token to start using the model.")