/** A symbol on screen, with its plain meaning one hover away.
 *
 * Deliberately not a replacement of the notation: the paper's symbols stay visible so a reader who
 * knows them is not made to relearn anything, and everyone else gets the sentence.
 */
import { GLOSSARY } from "./glossary";

export function Term({ id, children }: { id: keyof typeof GLOSSARY; children: React.ReactNode }) {
  const e = GLOSSARY[id];
  const tip = e.good ? `${e.name} — ${e.what} (${e.good})` : `${e.name} — ${e.what}`;
  return <abbr className="term" title={tip}>{children}</abbr>;
}

/** The plain-language name on its own, for places where the symbol would only get in the way. */
export function termName(id: keyof typeof GLOSSARY): string {
  return GLOSSARY[id].name;
}
