/**
 * React hooks for workspace data fetching
 */

'use client';

import { useState, useEffect, useCallback } from 'react';
import { getWorkspaces, createWorkspace, deleteWorkspace } from '@/lib/api';
import type { Workspace, WorkspaceCreateRequest } from '@/lib/types';

interface UseWorkspacesReturn {
  workspaces: Workspace[];
  total: number;
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  removeWorkspace: (id: string) => void;
}

/**
 * Hook to fetch and manage workspaces list
 */
export function useWorkspaces(): UseWorkspacesReturn {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchWorkspaces = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await getWorkspaces();
      setWorkspaces(response.workspaces);
      setTotal(response.total);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch workspaces';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWorkspaces();
  }, [fetchWorkspaces]);

  const removeWorkspace = useCallback((id: string) => {
    setWorkspaces((prev) => prev.filter((w) => w.id !== id));
    setTotal((prev) => Math.max(0, prev - 1));
  }, []);

  return {
    workspaces,
    total,
    isLoading,
    error,
    refresh: fetchWorkspaces,
    removeWorkspace,
  };
}

interface UseCreateWorkspaceReturn {
  createWorkspace: (data: WorkspaceCreateRequest) => Promise<Workspace | null>;
  isCreating: boolean;
  error: string | null;
  clearError: () => void;
}

/**
 * Hook to create a new workspace
 */
export function useCreateWorkspace(): UseCreateWorkspaceReturn {
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const create = useCallback(async (data: WorkspaceCreateRequest): Promise<Workspace | null> => {
    setIsCreating(true);
    setError(null);
    
    try {
      const workspace = await createWorkspace(data);
      return workspace;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create workspace';
      setError(message);
      return null;
    } finally {
      setIsCreating(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    createWorkspace: create,
    isCreating,
    error,
    clearError,
  };
}

interface UseDeleteWorkspaceReturn {
  deleteWorkspace: (id: string) => Promise<boolean>;
  isDeleting: boolean;
  error: string | null;
  clearError: () => void;
}

/**
 * Hook to delete a workspace
 */
export function useDeleteWorkspace(): UseDeleteWorkspaceReturn {
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const deleteWorkspaceFn = useCallback(async (id: string): Promise<boolean> => {
    setIsDeleting(true);
    setError(null);
    
    try {
      await deleteWorkspace(id);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete workspace';
      setError(message);
      return false;
    } finally {
      setIsDeleting(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    deleteWorkspace: deleteWorkspaceFn,
    isDeleting,
    error,
    clearError,
  };
}
