/**
 * Settings Page
 * 
 * Story 1.4: LLM Provider Configuration (Ollama & API Keys)
 * 
 * Allows users to configure application settings including
 * LLM provider configurations.
 */

'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LLMConfigForm } from '@/components/settings/LLMConfigForm';
import { EmbeddingConfigForm } from '@/components/settings/EmbeddingConfigForm';
import { ArrowLeft, Bot, Settings2, Layers } from 'lucide-react';

export default function SettingsPage() {
  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <Link href="/">
            <Button variant="ghost" size="sm" className="mb-4 -ml-2">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Dashboard
            </Button>
          </Link>
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-zinc-900 dark:bg-zinc-100">
              <Settings2 className="h-6 w-6 text-zinc-50 dark:text-zinc-900" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
                Settings
              </h1>
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                Configure your RAGitect preferences
              </p>
            </div>
          </div>
        </div>

        {/* Settings Tabs */}
        <Tabs defaultValue="llm" className="space-y-6">
          <TabsList className="grid w-full grid-cols-1 sm:w-auto sm:grid-cols-3">
            <TabsTrigger value="llm" className="flex items-center gap-2">
              <Bot className="h-4 w-4" />
              Chat Model
            </TabsTrigger>
            <TabsTrigger value="embedding" className="flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Embedding Model
            </TabsTrigger>
            <TabsTrigger value="general" className="flex items-center gap-2" disabled>
              <Settings2 className="h-4 w-4" />
              General
            </TabsTrigger>
          </TabsList>

          <TabsContent value="llm" className="space-y-4">
            <div className="mb-6">
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
                Chat Model Configuration
              </h2>
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                Configure your preferred AI models for chat and RAG tasks. Enable one or more providers 
                and save your settings. Your API keys are encrypted and stored securely.
              </p>
            </div>
            <LLMConfigForm />
          </TabsContent>

          <TabsContent value="embedding" className="space-y-4">
            <div className="mb-6">
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
                Embedding Model Configuration
              </h2>
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                Configure your preferred embedding models for document processing and semantic search. 
                Your API keys are encrypted and stored securely.
              </p>
            </div>
            <EmbeddingConfigForm />
          </TabsContent>

          <TabsContent value="general">
            <div className="rounded-lg border border-dashed border-zinc-300 bg-zinc-100/50 p-12 text-center dark:border-zinc-700 dark:bg-zinc-900/50">
              <h3 className="text-lg font-medium text-zinc-900 dark:text-zinc-50">
                General Settings
              </h3>
              <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
                Coming soon in a future release.
              </p>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
