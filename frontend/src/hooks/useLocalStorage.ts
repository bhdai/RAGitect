/**
 * useLocalStorage hook
 * 
 * Persist state to localStorage with SSR safety.
 */

'use client';

import { useState, useEffect, useRef } from 'react';

/**
 * Custom hook that syncs state with localStorage.
 * Handles SSR by defaulting to initial value on server.
 * 
 * @param key - The localStorage key
 * @param initialValue - Default value when key doesn't exist
 * @returns [value, setValue] tuple like useState
 */
export function useLocalStorage<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((prev: T) => T)) => void] {
  // Initialize with a function to read from localStorage only once
  const [storedValue, setStoredValue] = useState<T>(() => {
    // SSR safety: return initialValue if window is undefined
    if (typeof window === 'undefined') {
      return initialValue;
    }
    
    try {
      const item = window.localStorage.getItem(key);
      return item !== null ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.warn(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });
  
  // Use ref to track if we're past initial mount (avoids setState in effect)
  const isInitialMount = useRef(true);

  // Update localStorage when value changes (skip initial mount)
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }
    
    try {
      window.localStorage.setItem(key, JSON.stringify(storedValue));
    } catch (error) {
      console.warn(`Error setting localStorage key "${key}":`, error);
    }
  }, [key, storedValue]);

  const setValue = (value: T | ((prev: T) => T)) => {
    setStoredValue((prev) => {
      const newValue = value instanceof Function ? value(prev) : value;
      return newValue;
    });
  };

  return [storedValue, setValue];
}
