import { useEffect, useState } from "react";

/** `value`, but only after it has stopped changing for `ms`.
 *
 * Dragging the alpha slider fired one rollout per pointer move: a 764 ms drag sent 21 requests and
 * left 4.66 s of GPU work queued behind the single device lock after the user let go, 20 results of
 * which were discarded on arrival. Debouncing the *inputs* rather than throttling the requests keeps
 * the final position authoritative — the last value always runs.
 */
export function useDebounced<T>(value: T, ms = 120): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return settled;
}
