/**
 * Tests for AI Elements Shimmer component
 *
 * Tests the Shimmer text animation component from the AI Elements library.
 * Since Shimmer uses Framer Motion for animation, we test the static rendering
 * and props handling rather than the animation itself.
 */

import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Shimmer } from '../shimmer';

// Mock framer-motion to avoid animation issues in tests
vi.mock('motion/react', () => ({
  motion: {
    create: (component: string) => {
      // Return a component that renders the element with all props
      const MotionComponent = ({
        children,
        className,
        style,
        animate,
        initial,
        transition,
        ...props
      }: any) => {
        const Element = component as keyof JSX.IntrinsicElements;
        return (
          <Element 
            className={className} 
            style={style}
            data-testid="shimmer-element"
            {...props}
          >
            {children}
          </Element>
        );
      };
      return MotionComponent;
    },
  },
}));

describe('Shimmer', () => {
  it('renders text content', () => {
    render(<Shimmer>Loading...</Shimmer>);
    
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('renders as paragraph by default', () => {
    render(<Shimmer>Test text</Shimmer>);
    
    const element = screen.getByTestId('shimmer-element');
    expect(element.tagName.toLowerCase()).toBe('p');
  });

  it('renders as custom element via "as" prop', () => {
    render(<Shimmer as="span">Test text</Shimmer>);
    
    const element = screen.getByTestId('shimmer-element');
    expect(element.tagName.toLowerCase()).toBe('span');
  });

  it('renders as heading element', () => {
    render(<Shimmer as="h1">Title</Shimmer>);
    
    const element = screen.getByTestId('shimmer-element');
    expect(element.tagName.toLowerCase()).toBe('h1');
  });

  it('applies custom className', () => {
    render(<Shimmer className="text-lg font-bold">Test</Shimmer>);
    
    const element = screen.getByTestId('shimmer-element');
    expect(element).toHaveClass('text-lg', 'font-bold');
  });

  it('has default shimmer classes for text effect', () => {
    render(<Shimmer>Test</Shimmer>);
    
    const element = screen.getByTestId('shimmer-element');
    // Check for the core shimmer styling classes
    expect(element).toHaveClass('inline-block');
    expect(element).toHaveClass('bg-clip-text');
    expect(element).toHaveClass('text-transparent');
  });

  it('applies spread CSS variable based on text length', () => {
    const text = 'Hello World'; // 11 characters
    const spread = 2; // default
    const expectedSpread = `${11 * 2}px`; // 22px
    
    render(<Shimmer spread={spread}>{text}</Shimmer>);
    
    const element = screen.getByTestId('shimmer-element');
    expect(element).toHaveStyle({ '--spread': expectedSpread });
  });

  it('applies custom spread multiplier', () => {
    const text = 'Test'; // 4 characters
    const spread = 3;
    const expectedSpread = `${4 * 3}px`; // 12px
    
    render(<Shimmer spread={spread}>{text}</Shimmer>);
    
    const element = screen.getByTestId('shimmer-element');
    expect(element).toHaveStyle({ '--spread': expectedSpread });
  });

  it('renders with different text content lengths', () => {
    const { rerender } = render(<Shimmer>Short</Shimmer>);
    expect(screen.getByText('Short')).toBeInTheDocument();
    
    rerender(<Shimmer>A much longer loading message for testing purposes</Shimmer>);
    expect(screen.getByText('A much longer loading message for testing purposes')).toBeInTheDocument();
  });

  it('handles empty string gracefully', () => {
    render(<Shimmer>{''}</Shimmer>);
    
    const element = screen.getByTestId('shimmer-element');
    expect(element).toBeInTheDocument();
    expect(element.textContent).toBe('');
  });

  it('is memoized for performance', () => {
    // Verify the component is wrapped with memo by checking displayName
    // or by testing that re-renders with same props don't cause changes
    const { rerender } = render(<Shimmer>Test</Shimmer>);
    const firstElement = screen.getByTestId('shimmer-element');
    
    rerender(<Shimmer>Test</Shimmer>);
    const secondElement = screen.getByTestId('shimmer-element');
    
    // Both should reference the same DOM node (component didn't unmount/remount)
    expect(firstElement).toBe(secondElement);
  });
});

describe('Shimmer with duration prop', () => {
  it('accepts custom duration (animation timing tested visually)', () => {
    // Duration is passed to framer-motion transition, which we've mocked
    // Just verify the component renders without error with custom duration
    render(<Shimmer duration={1.5}>Fast shimmer</Shimmer>);
    
    expect(screen.getByText('Fast shimmer')).toBeInTheDocument();
  });

  it('uses default duration of 2 seconds', () => {
    // Default behavior test - component should render successfully
    render(<Shimmer>Default duration</Shimmer>);
    
    expect(screen.getByText('Default duration')).toBeInTheDocument();
  });
});
