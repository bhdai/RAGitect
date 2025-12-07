# RAGitect Frontend

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Testing

This project uses a dual-layer testing strategy:

### Unit & Component Tests (Vitest)

We use **Vitest** with **React Testing Library** for testing individual components, hooks, and utility functions.

**Run unit tests:**
```bash
npm run test              # Run tests once
npm run test:watch        # Run tests in watch mode
npm run test:ui           # Run tests with UI
```

**Test location:** `src/**/__tests__/*.test.tsx`

**What to test with Vitest:**
- Component rendering and props
- User interactions (clicks, keyboard input)
- Hook behavior
- Utility functions
- Client-side validation

### End-to-End Tests (Playwright)

We use **Playwright** for testing full user flows and page navigation.

**Run E2E tests:**
```bash
npm run test:e2e          # Run E2E tests
npm run test:e2e:ui       # Run E2E tests with UI
```

**Note:** E2E tests require the backend server to be running. Make sure to start the backend before running E2E tests.

**Test location:** `tests/e2e/*.spec.ts`

**What to test with Playwright:**
- Complete user journeys
- Page navigation
- Real API integration
- Form submissions and data persistence
- Visual regression (layout shifts)

### Testing Best Practices

- **Write tests first** (red-green-refactor cycle)
- **Unit tests** should be fast and isolated
- **E2E tests** should cover critical user paths
- **Mock external dependencies** in unit tests
- **Use real API** in E2E tests

For more details, see `docs/dev_notes/frontend_testing_strategy.md`.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
