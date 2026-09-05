'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { formatName } from '@/lib/format';

export interface UserCardProps {
  userId: string;
  compact?: boolean;
  onSelect?: (id: string) => void;
}

export default function UserCard({ userId, compact = false, onSelect }: UserCardProps) {
  const [name, setName] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/users?id=${userId}`)
      .then((r) => r.json())
      .then((u) => {
        if (!cancelled) {
          setName(formatName(u));
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [userId]);

  const handleClick = useCallback(() => onSelect?.(userId), [onSelect, userId]);

  if (loading) return <span className="skeleton" />;

  return (
    <div className={compact ? 'card card--compact' : 'card'} onClick={handleClick}>
      <h3>{name}</h3>
      {!compact && <p className="card__meta">id: {userId}</p>}
    </div>
  );
}

export class UserCardError extends Error {
  constructor(public readonly userId: string, message: string) {
    super(message);
  }

  describe(): string {
    return `${this.userId}: ${this.message}`;
  }
}

export const AnonymousWrapper = () => <UserCard userId="0" />;
