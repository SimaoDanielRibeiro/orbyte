import { createFileRoute } from "@tanstack/react-router";

import { SaveScreen } from "@/features/save/save-screen";

export const Route = createFileRoute("/save/$runId")({
  component: SaveRun,
});

function SaveRun() {
  const { runId } = Route.useParams();
  return <SaveScreen key={runId} runId={runId} />;
}
