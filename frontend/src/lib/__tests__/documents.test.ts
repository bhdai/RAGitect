import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { uploadDocuments } from '../documents';

describe('documents API', () => {
  const originalFetch = global.fetch;
  const mockFetch = vi.fn();

  beforeEach(() => {
    global.fetch = mockFetch;
    vi.clearAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  describe('uploadDocuments', () => {
    it('uploads files using FormData', async () => {
      const workspaceId = 'test-workspace-id';
      const file1 = new File(['content1'], 'file1.txt', { type: 'text/plain' });
      const file2 = new File(['content2'], 'file2.md', { type: 'text/markdown' });
      const files = [file1, file2];

      const mockResponse = {
        documents: [
          {
            id: 'doc-1',
            fileName: 'file1.txt',
            fileType: '.txt',
            status: 'uploaded',
            createdAt: '2025-12-12T00:00:00Z',
          },
          {
            id: 'doc-2',
            fileName: 'file2.md',
            fileType: '.md',
            status: 'uploaded',
            createdAt: '2025-12-12T00:00:00Z',
          },
        ],
        total: 2,
      };

      mockFetch.mockResolvedValue({
        ok: true,
        status: 201,
        json: async () => mockResponse,
      });

      const result = await uploadDocuments(workspaceId, files);

      expect(mockFetch).toHaveBeenCalledTimes(1);
      
      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toContain(`/api/v1/workspaces/${workspaceId}/documents`);
      expect(options.method).toBe('POST');
      expect(options.body).toBeInstanceOf(FormData);
      
      expect(result).toEqual(mockResponse);
      expect(result.documents).toHaveLength(2);
      expect(result.documents[0].fileName).toBe('file1.txt');
    });

    it('throws error when workspace not found', async () => {
      const workspaceId = 'non-existent';
      const files = [new File(['test'], 'test.txt', { type: 'text/plain' })];

      mockFetch.mockResolvedValue({
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Workspace not found' }),
      });

      await expect(uploadDocuments(workspaceId, files)).rejects.toThrow('Workspace not found');
    });

    it('throws error for unsupported file types', async () => {
      const workspaceId = 'test-workspace';
      const files = [new File(['test'], 'test.exe', { type: 'application/x-msdownload' })];

      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          detail: 'Unsupported format: .exe',
        }),
      });

      await expect(uploadDocuments(workspaceId, files)).rejects.toThrow('Unsupported format');
    });

    it('handles network errors', async () => {
      const workspaceId = 'test-workspace';
      const files = [new File(['test'], 'test.txt', { type: 'text/plain' })];

      mockFetch.mockRejectedValue(new Error('Network error'));

      await expect(uploadDocuments(workspaceId, files)).rejects.toThrow('Network error');
    });

    it('includes all files in FormData', async () => {
      const workspaceId = 'test-workspace';
      const files = [
        new File(['content1'], 'file1.txt', { type: 'text/plain' }),
        new File(['content2'], 'file2.md', { type: 'text/markdown' }),
        new File(['content3'], 'file3.pdf', { type: 'application/pdf' }),
      ];

      mockFetch.mockResolvedValue({
        ok: true,
        status: 201,
        json: async () => ({
          documents: [],
          total: 0,
        }),
      });

      await uploadDocuments(workspaceId, files);

      const formData = mockFetch.mock.calls[0][1].body as FormData;
      const uploadedFiles = formData.getAll('files');
      
      expect(uploadedFiles).toHaveLength(3);
    });
  });
});
