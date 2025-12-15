/**
 * Chat panel placeholder for Story 3.1
 * 
 * Shows "Chat coming soon" message until full chat functionality
 * is implemented. Takes remaining space in the three-panel layout.
 * 
 * Story 3.0: Streaming Infrastructure - AC3
 */

'use client';

import { MessageSquare } from 'lucide-react';

export interface ChatPanelProps {
  /** Workspace ID for future chat operations */
  workspaceId: string;
}

/**
 * Placeholder chat panel for the workspace three-panel layout.
 * 
 * Will be implemented with full chat functionality in Story 3.1.
 * For now, displays a centered message indicating the feature is coming.
 */
export function ChatPanel({ workspaceId }: ChatPanelProps) {
  // workspaceId will be used in Story 3.1 for actual chat functionality
  void workspaceId;

  return (
    <div
      data-testid="chat-panel"
      className="flex-1 flex items-center justify-center bg-zinc-50 dark:bg-zinc-900 overflow-y-auto"
    >
      <div className="text-center text-muted-foreground">
        <MessageSquare className="mx-auto h-12 w-12 mb-4 opacity-50" />
        <p className="text-lg font-medium">Chat with your documents</p>
        <p className="text-sm mt-1">Coming in Story 3.1</p>
      </div>
    </div>
  );
}
