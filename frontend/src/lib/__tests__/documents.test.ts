import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { uploadDocuments, uploadUrl, detectUrlType } from '../documents';

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

  describe('uploadUrl', () => {
    it('sends POST request with JSON body', async () => {
      const workspaceId = 'test-workspace-id';
      const url = 'https://example.com/article';
      const sourceType = 'url' as const;

      const mockResponse = {
        id: 'doc-1',
        sourceType: 'url',
        sourceUrl: url,
        status: 'backlog',
        message: 'URL submitted for processing',
      };

      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => mockResponse,
      });

      const result = await uploadUrl(workspaceId, url, sourceType);

      expect(mockFetch).toHaveBeenCalledTimes(1);

      const [fetchUrl, options] = mockFetch.mock.calls[0];
      expect(fetchUrl).toContain(`/api/v1/workspaces/${workspaceId}/documents/upload-url`);
      expect(options.method).toBe('POST');
      expect(options.headers).toEqual({ 'Content-Type': 'application/json' });
      expect(JSON.parse(options.body)).toEqual({ sourceType: 'url', url });

      expect(result).toEqual(mockResponse);
    });

    it('sends correct sourceType for YouTube URLs', async () => {
      const mockResponse = {
        id: 'doc-2',
        sourceType: 'youtube',
        sourceUrl: 'https://youtube.com/watch?v=abc',
        status: 'backlog',
        message: 'URL submitted for processing',
      };

      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => mockResponse,
      });

      await uploadUrl('ws-1', 'https://youtube.com/watch?v=abc', 'youtube');

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.sourceType).toBe('youtube');
    });

    it('throws error on 400 invalid URL', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Only HTTP and HTTPS URLs are allowed' }),
      });

      await expect(uploadUrl('ws-1', 'ftp://example.com', 'url')).rejects.toThrow(
        'Only HTTP and HTTPS URLs are allowed'
      );
    });

    it('throws error on 409 duplicate URL', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 409,
        json: async () => ({ detail: 'URL already submitted for this workspace' }),
      });

      await expect(uploadUrl('ws-1', 'https://example.com', 'url')).rejects.toThrow(
        'URL already submitted for this workspace'
      );
    });

    it('handles network errors', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      await expect(uploadUrl('ws-1', 'https://example.com', 'url')).rejects.toThrow(
        'Network error'
      );
    });

    it('handles non-JSON error responses', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => { throw new Error('not json'); },
      });

      await expect(uploadUrl('ws-1', 'https://example.com', 'url')).rejects.toThrow(
        'URL upload failed with status 500'
      );
    });
  });

  describe('detectUrlType', () => {
    it('detects YouTube URLs with youtube.com/watch', () => {
      expect(detectUrlType('https://youtube.com/watch?v=abc123')).toBe('youtube');
      expect(detectUrlType('https://www.youtube.com/watch?v=abc123')).toBe('youtube');
    });

    it('detects YouTube URLs with youtu.be/', () => {
      expect(detectUrlType('https://youtu.be/abc123')).toBe('youtube');
    });

    it('detects YouTube URLs case-insensitively', () => {
      expect(detectUrlType('https://YOUTUBE.COM/watch?v=abc')).toBe('youtube');
      expect(detectUrlType('https://YouTu.Be/abc')).toBe('youtube');
    });

    it('detects PDF URLs by extension', () => {
      expect(detectUrlType('https://example.com/document.pdf')).toBe('pdf');
    });

    it('detects PDF URLs with query parameters', () => {
      expect(detectUrlType('https://example.com/doc.pdf?token=abc')).toBe('pdf');
    });

    it('detects PDF URLs case-insensitively', () => {
      expect(detectUrlType('https://example.com/Document.PDF')).toBe('pdf');
    });

    it('returns url for regular web URLs', () => {
      expect(detectUrlType('https://example.com/article')).toBe('url');
      expect(detectUrlType('https://blog.example.com/post/123')).toBe('url');
    });

    it('returns url for empty or whitespace input', () => {
      expect(detectUrlType('')).toBe('url');
      expect(detectUrlType('   ')).toBe('url');
    });

    it('handles URLs with trailing whitespace', () => {
      expect(detectUrlType('https://youtube.com/watch?v=abc  ')).toBe('youtube');
      expect(detectUrlType('https://example.com/doc.pdf  ')).toBe('pdf');
    });
  });
});
