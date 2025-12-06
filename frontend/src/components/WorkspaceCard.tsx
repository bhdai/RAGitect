/**
 * WorkspaceCard component
 * 
 * Displays a workspace in a Linear-style card with name, description,
 * and creation timestamp. Clicking navigates to the workspace detail page.
 */

'use client';

import { useRouter } from 'next/navigation';
import { 
  Card, 
  CardHeader, 
  CardTitle, 
  CardDescription, 
  CardContent 
} from '@/components/ui/card';
import type { Workspace } from '@/lib/types';

interface WorkspaceCardProps {
  workspace: Workspace;
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

export function WorkspaceCard({ workspace }: WorkspaceCardProps) {
  const router = useRouter();

  const handleClick = () => {
    router.push(`/workspace/${workspace.id}`);
  };

  return (
    <Card 
      className="cursor-pointer transition-all hover:shadow-md hover:border-zinc-300 dark:hover:border-zinc-600"
      onClick={handleClick}
    >
      <CardHeader>
        <CardTitle className="text-lg">{workspace.name}</CardTitle>
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
    </Card>
  );
}
