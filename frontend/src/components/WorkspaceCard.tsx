/**
 * WorkspaceCard component
 * 
 * Displays a workspace in a Linear-style card with name, description,
 * and creation timestamp. Clicking navigates to the workspace detail page.
 * Includes dropdown menu with delete option.
 */

'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { MoreVertical, Trash2 } from 'lucide-react';
import { 
  Card, 
  CardHeader, 
  CardTitle, 
  CardDescription, 
  CardContent 
} from '@/components/ui/card';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { DeleteWorkspaceDialog } from './DeleteWorkspaceDialog';
import { toast } from 'sonner';
import type { Workspace } from '@/lib/types';

interface WorkspaceCardProps {
  workspace: Workspace;
  onDelete?: (id: string) => Promise<void>;
}

/**
 * Format a date string to a readable format
 */
function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

export function WorkspaceCard({ workspace, onDelete }: WorkspaceCardProps) {
  const router = useRouter();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const handleCardClick = () => {
    router.push(`/workspace/${workspace.id}`);
  };

  const handleCardKeyDown = (e: React.KeyboardEvent) => {
    // Handle Enter and Space keys for accessibility
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleCardClick();
    }
  };

  const handleDeleteClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent card click
    setShowDeleteDialog(true);
  };

  const handleDeleteConfirm = async () => {
    if (!onDelete) {
      toast.error('Delete function not available');
      return;
    }

    try {
      await onDelete(workspace.id);
      // Close dialog after successful deletion
      setShowDeleteDialog(false);
      toast.success(`Workspace "${workspace.name}" has been deleted`);
    } catch (error) {
      console.error('Failed to delete workspace:', error);
      toast.error(
        error instanceof Error ? error.message : 'Failed to delete workspace. Please try again.'
      );
      // Keep dialog open on error so user can retry
    }
  };

  return (
    <>
      <Card 
        className="cursor-pointer transition-all hover:shadow-md hover:border-zinc-300 dark:hover:border-zinc-600 relative focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
        tabIndex={0}
        role="button"
        aria-label={`Open workspace ${workspace.name}`}
        onKeyDown={handleCardKeyDown}
      >
        <div onClick={handleCardClick} className="flex-1">
          <CardHeader>
            <div className="flex items-start justify-between">
              <CardTitle className="text-lg">{workspace.name}</CardTitle>
              <DropdownMenu>
                <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                  <Button 
                    variant="ghost" 
                    size="icon"
                    className="h-8 w-8 -mt-1 -mr-1"
                    aria-label={`More options for ${workspace.name}`}
                  >
                    <MoreVertical className="h-4 w-4" />
                    <span className="sr-only">Open menu</span>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem 
                    onClick={handleDeleteClick}
                    className="text-destructive focus:text-destructive"
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    Delete Workspace
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            {workspace.description && (
              <CardDescription className="line-clamp-2">
                {workspace.description}
              </CardDescription>
            )}
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              Created {formatDate(workspace.createdAt)}
            </p>
          </CardContent>
        </div>
      </Card>

      <DeleteWorkspaceDialog
        workspaceName={workspace.name}
        open={showDeleteDialog}
        onOpenChange={setShowDeleteDialog}
        onConfirm={handleDeleteConfirm}
      />
    </>
  );
}

