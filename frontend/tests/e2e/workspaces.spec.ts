/**
 * E2E Tests for Workspace Management
 * 
 * Tests complete user flows including workspace creation, deletion,
 * and UI state synchronization.
 */

import { test, expect } from '@playwright/test';

test.describe('Workspace Deletion', () => {
  test('should remove workspace from UI immediately after deletion', async ({ page }) => {
    // Navigate to the homepage
    await page.goto('/');
    
    // Wait for the page to load
    await expect(page.getByText('RAGitect')).toBeVisible();
    
    // Create a workspace for testing
    const testWorkspaceName = `Test Workspace ${Date.now()}`;
    
    // Click "New Workspace" button
    await page.getByRole('button', { name: /new workspace/i }).click();
    
    // Fill in the form
    await page.getByLabel(/name/i).fill(testWorkspaceName);
    await page.getByLabel(/description/i).fill('This workspace will be deleted');
    
    // Submit the form
    await page.getByRole('button', { name: /create/i }).click();
    
    // Wait for success notification
    await expect(page.getByText(/workspace created successfully/i)).toBeVisible();
    
    // Verify the workspace appears in the list
    await expect(page.getByText(testWorkspaceName)).toBeVisible();
    
    // Now delete the workspace
    // Find the workspace card and click the more options button
    const workspaceCard = page.locator('[role="button"]', { hasText: testWorkspaceName });
    const moreButton = workspaceCard.locator('button[aria-haspopup="menu"]');
    await moreButton.click();
    
    // Click "Delete Workspace" in the dropdown
    await page.getByText('Delete Workspace').click();
    
    // Confirm deletion in the dialog
    await page.getByRole('button', { name: /delete/i }).click();
    
    // Wait for success notification
    await expect(page.getByText(/has been deleted/i)).toBeVisible();
    
    // CRITICAL ASSERTION: The workspace should NOT be visible anymore
    // This test is expected to FAIL before the bug fix
    await expect(page.getByText(testWorkspaceName)).not.toBeVisible({ timeout: 2000 });
  });

  test('should handle deletion errors gracefully', async ({ page }) => {
    // This test verifies error handling when API fails
    // We'll mock a failed deletion by intercepting the API call
    
    await page.route('**/api/v1/workspaces/*', async (route) => {
      if (route.request().method() === 'DELETE') {
        await route.fulfill({
          status: 500,
          body: JSON.stringify({ detail: 'Internal server error' }),
        });
      } else {
        await route.continue();
      }
    });
    
    await page.goto('/');
    
    // Assume there's at least one workspace (or create one first)
    // For this test, we'll try to delete the first available workspace
    const firstWorkspace = page.locator('[role="button"][aria-label^="Open workspace"]').first();
    
    if (await firstWorkspace.count() > 0) {
      const moreButton = firstWorkspace.locator('button[aria-haspopup="menu"]');
      await moreButton.click();
      
      await page.getByText('Delete Workspace').click();
      await page.getByRole('button', { name: /delete/i }).click();
      
      // Should show error notification
      await expect(page.getByText(/failed to delete/i)).toBeVisible();
      
      // Workspace should still be visible since deletion failed
      await expect(firstWorkspace).toBeVisible();
    }
  });
});
