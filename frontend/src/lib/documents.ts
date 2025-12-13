/**
 * Documents API client
 * 
 * Handles document upload, listing, detail, and delete operations.
 */

import type { ApiError } from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Document {
  id: string;
  fileName: string;
  fileType: string | null;
  status: string;
  createdAt: string;
}

export interface DocumentDetail {
  id: string;
  fileName: string;
  fileType: string | null;
  status: string;
  processedContent: string | null;
  summary: string | null;
  createdAt: string;
}

export interface DocumentUploadResponse {
  documents: Document[];
  total: number;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
}

export interface DocumentStatus {
  id: string;
  status: string;
  fileName: string;
  phase: 'parsing' | 'embedding' | null; // Current processing phase
}

/**
 * Upload multiple documents to a workspace
 * 
 * @param workspaceId - Target workspace ID
 * @param files - Array of File objects to upload
 * @returns Promise with upload response containing document metadata
 * @throws Error if upload fails
 */
export async function uploadDocuments(
  workspaceId: string,
  files: File[]
): Promise<DocumentUploadResponse> {
  const url = `${API_BASE}/api/v1/workspaces/${workspaceId}/documents`;

  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  const response = await fetch(url, {
    method: 'POST',
    body: formData,
    // Don't set Content-Type header - browser will set it with boundary
  });

  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({
      detail: `Upload failed with status ${response.status}`,
    }));
    throw new Error(error.detail);
  }

  return response.json();
}

/**
 * Get document processing status
 * 
 * @param documentId - Document ID to check
 * @returns Promise with document status
 * @throws Error if request fails
 */
export async function getDocumentStatus(
  documentId: string
): Promise<DocumentStatus> {
  const url = `${API_BASE}/api/v1/workspaces/documents/${documentId}/status`;

  const response = await fetch(url, {
    method: 'GET',
  });

  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({
      detail: `Status fetch failed with status ${response.status}`,
    }));
    throw new Error(error.detail);
  }

  return response.json();
}

/**
 * Get all documents in a workspace
 * 
 * @param workspaceId - Workspace ID to list documents for
 * @returns Promise with document list response
 * @throws Error if request fails
 */
export async function getDocuments(
  workspaceId: string
): Promise<DocumentListResponse> {
  const url = `${API_BASE}/api/v1/workspaces/${workspaceId}/documents`;

  const response = await fetch(url, {
    method: 'GET',
  });

  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({
      detail: `Failed to fetch documents with status ${response.status}`,
    }));
    throw new Error(error.detail);
  }

  return response.json();
}

/**
 * Get document detail with full content
 * 
 * @param documentId - Document ID to fetch
 * @returns Promise with document detail including processed content
 * @throws Error if request fails
 */
export async function getDocument(
  documentId: string
): Promise<DocumentDetail> {
  const url = `${API_BASE}/api/v1/documents/${documentId}`;

  const response = await fetch(url, {
    method: 'GET',
  });

  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({
      detail: `Failed to fetch document with status ${response.status}`,
    }));
    throw new Error(error.detail);
  }

  return response.json();
}

/**
 * Delete a document
 * 
 * @param documentId - Document ID to delete
 * @throws Error if request fails
 */
export async function deleteDocument(documentId: string): Promise<void> {
  const url = `${API_BASE}/api/v1/documents/${documentId}`;

  const response = await fetch(url, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({
      detail: `Failed to delete document with status ${response.status}`,
    }));
    throw new Error(error.detail);
  }
}
