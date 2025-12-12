/**
 * Documents API client
 * 
 * Handles document upload operations with multipart/form-data.
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

export interface DocumentUploadResponse {
  documents: Document[];
  total: number;
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
