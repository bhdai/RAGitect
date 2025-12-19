'use client';

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useProviderSelectionContext } from '@/contexts/ProviderSelectionContext';
import { AlertCircle, Loader2, Settings } from 'lucide-react';
import Link from 'next/link';

export function ChatProviderSelector() {
  const {
    selectedProvider,
    selectProvider,
    providers,
    currentProvider,
    isLoading,
    error
  } = useProviderSelectionContext();

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 px-2 text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div 
        className="flex items-center gap-2 px-2 text-destructive text-xs"
        title={error.message}
      >
        <AlertCircle className="h-4 w-4" />
        <span>Failed to load models</span>
      </div>
    );
  }

  if (providers.length === 0) {
    return (
      <Link
        href="/settings"
        className="flex items-center gap-2 px-2 text-sm text-muted-foreground hover:text-foreground"
      >
        <Settings className="h-4 w-4" />
        Configure model
      </Link>
    );
  }

  return (
    <Select value={selectedProvider || ''} onValueChange={selectProvider}>
      <SelectTrigger size="sm" className="!h-7 w-auto gap-1 border-0 bg-transparent px-2 text-xs font-medium hover:bg-accent focus:ring-0">
        <SelectValue>
          {currentProvider?.model || 'Select model'}
        </SelectValue>
      </SelectTrigger>
      <SelectContent align="start">
        {providers.map((provider) => (
          <SelectItem
            key={provider.providerName}
            value={provider.providerName}
            className="text-sm"
          >
            <div className="flex items-center gap-2">
              <span className="font-medium">{provider.model}</span>
              <span className="text-muted-foreground text-xs">
                ({provider.displayName})
              </span>
            </div>
          </SelectItem>
        ))}
        <div className="border-t mt-1 pt-1">
          <Link
            href="/settings"
            className="flex items-center gap-2 px-2 py-1.5 text-sm text-muted-foreground hover:text-foreground"
          >
            <Settings className="h-3 w-3" />
            Configure models...
          </Link>
        </div>
      </SelectContent>
    </Select>
  );
}
