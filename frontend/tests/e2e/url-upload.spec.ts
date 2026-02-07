/**
 * E2E Tests for URL Upload Flow
 *
 * Tests URL submission via modal with mocked API responses,
 * document list integration, and error scenarios.
 */

import { test, expect } from '@playwright/test';

/**
 * Helper to set up API mocks for workspace and document operations
 */
async function setupWorkspaceMocks(page: import('@playwright/test').Page, workspaceId: string) {
  // Mock workspace detail
  await page.route(`**/api/v1/workspaces/${workspaceId}`, async (route) => {
    if (route.request().method() === 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: workspaceId,
          name: 'Test Workspace',
          description: 'A test workspace',
          createdAt: '2025-12-01T00:00:00Z',
          updatedAt: '2025-12-01T00:00:00Z',
        }),
      });
    } else {
      await route.continue();
    }
  });

  // Mock document list
  await page.route(`**/api/v1/workspaces/${workspaceId}/documents`, async (route) => {
    if (route.request().method() === 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ documents: [], total: 0 }),
      });
    } else {
      await route.continue();
    }
  });

  // Mock chat providers
  await page.route('**/api/v1/chat/providers', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ providers: [] }),
    });
  });
}

test.describe('URL Upload', () => {
  const workspaceId = 'test-ws-url';

  test('should submit a web URL via modal and show in progress', async ({ page }) => {
    await setupWorkspaceMocks(page, workspaceId);

    // Mock URL upload endpoint
    await page.route(`**/api/v1/workspaces/${workspaceId}/documents/upload-url`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'doc-url-1',
          sourceType: 'url',
          sourceUrl: 'https://example.com/article',
          status: 'backlog',
          message: 'URL submitted for processing',
        }),
      });
    });

    // Mock status polling — first call returns fetching, second returns ready
    let pollCount = 0;
    await page.route('**/api/v1/workspaces/documents/doc-url-1/status', async (route) => {
      pollCount++;
      if (pollCount <= 1) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 'doc-url-1',
            status: 'fetching',
            fileName: 'https://example.com/article',
            phase: null,
          }),
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 'doc-url-1',
            status: 'ready',
            fileName: 'https://example.com/article',
            phase: null,
          }),
        });
      }
    });

    await page.goto(`/workspace/${workspaceId}`);
    await expect(page.getByText('Test Workspace')).toBeVisible();

    // Open the upload modal
    await page.getByTestId('add-source-button').click();
    await expect(page.getByTestId('upload-modal')).toBeVisible();

    // Enter URL
    const urlInput = page.getByTestId('url-input');
    await urlInput.fill('https://example.com/article');

    // Should show web icon
    await expect(page.getByTestId('url-type-icon-web')).toBeVisible();

    // Click Add URL
    await page.getByTestId('add-url-button').click();

    // URL input should be cleared
    await expect(urlInput).toHaveValue('');

    // Toast should show
    await expect(page.getByText(/URL submitted/i)).toBeVisible();
  });

  test('should show YouTube icon for YouTube URLs in modal', async ({ page }) => {
    await setupWorkspaceMocks(page, workspaceId);

    await page.goto(`/workspace/${workspaceId}`);
    await page.getByTestId('add-source-button').click();
    await expect(page.getByTestId('upload-modal')).toBeVisible();

    await page.getByTestId('url-input').fill('https://youtube.com/watch?v=abc123');
    await expect(page.getByTestId('url-type-icon-youtube')).toBeVisible();
  });

  test('should show PDF icon for PDF URLs in modal', async ({ page }) => {
    await setupWorkspaceMocks(page, workspaceId);

    await page.goto(`/workspace/${workspaceId}`);
    await page.getByTestId('add-source-button').click();
    await expect(page.getByTestId('upload-modal')).toBeVisible();

    await page.getByTestId('url-input').fill('https://example.com/document.pdf');
    await expect(page.getByTestId('url-type-icon-pdf')).toBeVisible();
  });

  test('should display error for invalid URL (400)', async ({ page }) => {
    await setupWorkspaceMocks(page, workspaceId);

    // Mock 400 error
    await page.route(`**/api/v1/workspaces/${workspaceId}/documents/upload-url`, async (route) => {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Only HTTP and HTTPS URLs are allowed',
        }),
      });
    });

    await page.goto(`/workspace/${workspaceId}`);
    await page.getByTestId('add-source-button').click();
    await page.getByTestId('url-input').fill('ftp://example.com/file');
    await page.getByTestId('add-url-button').click();

    // Error toast should appear
    await expect(page.getByText(/Only HTTP and HTTPS URLs are allowed/i)).toBeVisible();
  });

  test('should display error for SSRF-blocked URL', async ({ page }) => {
    await setupWorkspaceMocks(page, workspaceId);

    await page.route(`**/api/v1/workspaces/${workspaceId}/documents/upload-url`, async (route) => {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Private and localhost URLs are not allowed',
        }),
      });
    });

    await page.goto(`/workspace/${workspaceId}`);
    await page.getByTestId('add-source-button').click();
    await page.getByTestId('url-input').fill('http://localhost:8080/secret');
    await page.getByTestId('add-url-button').click();

    await expect(page.getByText(/Private and localhost URLs are not allowed/i)).toBeVisible();
  });

  test('should display error for duplicate URL (409)', async ({ page }) => {
    await setupWorkspaceMocks(page, workspaceId);

    await page.route(`**/api/v1/workspaces/${workspaceId}/documents/upload-url`, async (route) => {
      await route.fulfill({
        status: 409,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'URL already submitted for this workspace',
        }),
      });
    });

    await page.goto(`/workspace/${workspaceId}`);
    await page.getByTestId('add-source-button').click();
    await page.getByTestId('url-input').fill('https://example.com/article');
    await page.getByTestId('add-url-button').click();

    await expect(page.getByText(/URL already submitted for this workspace/i)).toBeVisible();
  });

  test('should show document with correct source-type icon after ingestion', async ({ page }) => {
    const wsId = 'test-ws-icons';

    // Mock workspace
    await page.route(`**/api/v1/workspaces/${wsId}`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: wsId,
          name: 'Icon Test Workspace',
          description: null,
          createdAt: '2025-12-01T00:00:00Z',
          updatedAt: '2025-12-01T00:00:00Z',
        }),
      });
    });

    // Mock document list with URL-based documents
    await page.route(`**/api/v1/workspaces/${wsId}/documents`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          documents: [
            { id: 'd1', fileName: 'https://youtube.com/watch?v=abc', fileType: null, status: 'ready', createdAt: '2025-12-01T00:00:00Z' },
            { id: 'd2', fileName: 'https://example.com/paper.pdf', fileType: null, status: 'ready', createdAt: '2025-12-01T00:00:00Z' },
            { id: 'd3', fileName: 'https://blog.example.com/post', fileType: null, status: 'ready', createdAt: '2025-12-01T00:00:00Z' },
            { id: 'd4', fileName: 'report.docx', fileType: 'docx', status: 'ready', createdAt: '2025-12-01T00:00:00Z' },
          ],
          total: 4,
        }),
      });
    });

    // Mock chat providers
    await page.route('**/api/v1/chat/providers', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ providers: [] }),
      });
    });

    await page.goto(`/workspace/${wsId}`);

    // Verify icons render for each document type
    await expect(page.getByTestId('doc-icon-youtube')).toBeVisible();
    await expect(page.getByTestId('doc-icon-pdf')).toBeVisible();
    await expect(page.getByTestId('doc-icon-globe')).toBeVisible();
    await expect(page.getByTestId('doc-icon-file')).toBeVisible();
  });
});
