/**
 * The telemetry subscription.
 *
 * Split out from `api.ts` because the connection *state* is as important as the
 * data. A dashboard whose socket dies while the last frame stays on screen is
 * actively dangerous — the numbers look authoritative while being arbitrarily
 * stale, and VRAM headroom is exactly the kind of figure someone acts on.
 *
 * So this hook reports three things, and the UI is expected to show all of them:
 * the latest frame, whether the stream is currently live, and how long ago the
 * last frame arrived.
 */

import { useEffect, useRef, useState } from "react";
import type { Telemetry } from "./api";

export type ConnectionState = "connecting" | "live" | "lost";

export interface TelemetryFeed {
  data: Telemetry | null;
  connection: ConnectionState;
  /** Seconds since the last frame. Grows visibly when the feed stalls. */
  staleness: number;
  error: string | null;
}

/** A frame older than this is treated as stale even if the socket looks open. */
const STALE_AFTER_SECONDS = 5;

export function useTelemetry(url = "/api/telemetry"): TelemetryFeed {
  const [data, setData] = useState<Telemetry | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [error, setError] = useState<string | null>(null);
  const [staleness, setStaleness] = useState(0);
  const lastFrame = useRef<number>(Date.now());

  useEffect(() => {
    const source = new EventSource(url);

    source.onopen = () => {
      setConnection("live");
      setError(null);
    };

    source.onmessage = (event: MessageEvent<string>) => {
      try {
        setData(JSON.parse(event.data) as Telemetry);
        lastFrame.current = Date.now();
        setConnection("live");
        setError(null);
      } catch (err) {
        // A malformed frame is a bug worth surfacing, but it must not tear down
        // a stream that is otherwise delivering.
        setError(err instanceof Error ? err.message : "malformed telemetry frame");
      }
    };

    source.onerror = () => {
      // EventSource reconnects on its own, so this is "lost", not "failed".
      // Saying "failed" would imply the user has to do something.
      setConnection("lost");
    };

    return () => source.close();
  }, [url]);

  // Track staleness independently of the stream. If frames stop arriving while
  // the socket stays nominally open -- a hung backend, a suspended laptop -- the
  // socket state alone would keep claiming everything is fine.
  useEffect(() => {
    const timer = window.setInterval(() => {
      const age = (Date.now() - lastFrame.current) / 1000;
      setStaleness(age);
      if (age > STALE_AFTER_SECONDS) {
        setConnection((current) => (current === "live" ? "lost" : current));
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  return { data, connection, staleness, error };
}
