/**
 * API client for RAGitect backend
 * 
 * Uses native fetch with proper error handling for workspace operations.
 * 
 * IMPORTANT: NEXT_PUBLIC_API_URL must be set to a URL reachable from the browser.
 * In Docker, this means http://localhost:8000 (the host-exposed port), NOT
 * http://backend:8000 (internal Docker network).
 */

import type { 
  Workspace, 
  WorkspaceListResponse, 
  WorkspaceCreateRequest,
  ApiError 
} from './types';

// Client-side API calls run in the browser, which reaches localhost:8000
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({
        detail: `Request failed with status ${response.status}`,
      }));
      throw new Error(error.detail);
    }

    return response.json();
  }

  /**
   * Fetch all workspaces
   */
  async getWorkspaces(): Promise<WorkspaceListResponse> {
    return this.request<WorkspaceListResponse>('/api/v1/workspaces');
  }

  /**
   * Fetch a single workspace by ID
   */
  async getWorkspace(id: string): Promise<Workspace> {
    return this.request<Workspace>(`/api/v1/workspaces/${id}`);
  }

  /**
   * Create a new workspace
   */
  async createWorkspace(data: WorkspaceCreateRequest): Promise<Workspace> {
    return this.request<Workspace>('/api/v1/workspaces', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /**
   * Delete a workspace by ID
   */
  async deleteWorkspace(id: string): Promise<void> {
    const url = `${this.baseUrl}/api/v1/workspaces/${id}`;
    
    const response = await fetch(url, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (response.status === 404) {
      throw new Error('Workspace not found');
    }

    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({
        detail: `Failed to delete workspace: ${response.status}`,
      }));
      throw new Error(error.detail);
    }

    // 204 No Content - no response body to parse
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Export individual functions for convenience
export const getWorkspaces = () => apiClient.getWorkspaces();
export const getWorkspace = (id: string) => apiClient.getWorkspace(id);
export const createWorkspace = (data: WorkspaceCreateRequest) => 
  apiClient.createWorkspace(data);
export const deleteWorkspace = (id: string) => apiClient.deleteWorkspace(id);
