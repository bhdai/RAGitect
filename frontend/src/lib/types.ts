/**
 * TypeScript types for Workspace API
 * 
 * Note: These match the camelCase serialization from the backend API
 */

export interface Workspace {
  id: string;
  name: string;
  description: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface WorkspaceListResponse {
  workspaces: Workspace[];
  total: number;
}

export interface WorkspaceCreateRequest {
  name: string;
  description?: string;
}

export interface ApiError {
  detail: string;
}
