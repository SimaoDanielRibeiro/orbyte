/**
 * A biblioteca dos modelos guardados, e o carregar de um modelo já treinado (ADR 0039).
 *
 * Os pacotes ficam numa lista, uma linha cada (ADR 0051): a marca da cor da família do modelo
 * (ADR 0038), o nome, o modelo e os dados, a métrica principal, o tamanho, a data e de onde veio,
 * e as ações. Abre-se ao lado, no mesmo leitor da pré-visualização, descarrega-se ou apaga-se.
 * Por cima da lista, uma faixa recebe um `.mllab` largado ou escolhido.
 *
 * Carregar um `.mllab` põe-no na biblioteca. Um pacote feito noutro ML Lab mostra logo o aviso
 * (o texto é o do backend): a ficha, a avaliação e a preparação leem-se à vontade, e o modelo só
 * abre depois de a pessoa confirmar que confia em quem lhe deu o ficheiro.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { cn } from "cn";
import { Download, FileArchive, FileUp, LoaderCircle, Trash2 } from "lucide-react";
import { useState, type DragEvent } from "react";
import { toast } from "sonner";

import type { SavedPackage } from "@/api/generated";
import {
  packagesContentsOptions,
  packagesDeleteMutation,
  packagesLoadMutation,
  packagesSavedOptions,
  packagesTrustMutation,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, Section, StatStrip } from "@/components/shell/blocks";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { OTHER } from "@/lib/colors";
import { familyColor } from "@/lib/families";
import { formatBytes, formatMoment, formatNumber, formatPercent } from "@/lib/format";

import { invalidatePackages, packageFileUrl } from "./cache";
import { PackageViewer } from "./package-viewer";

/** O aviso antes de abrir um modelo de outro ML Lab, e se o leitor abre quando ele fechar. */
interface Warning {
  pkg: SavedPackage;
  open: boolean;
  /** Abrir o pacote no leitor depois do aviso (um pacote acabado de carregar). */
  thenRead: boolean;
}

export function Library() {
  const saved = useQuery(packagesSavedOptions());
  const [opened, setOpened] = useState<string | null>(null);
  // O pacote fica no aviso enquanto ele se fecha (o texto não desaparece a meio da animação).
  const [warning, setWarning] = useState<Warning | null>(null);
  const ask = (pkg: SavedPackage) => {
    setWarning({ pkg, open: true, thenRead: false });
  };
  const count = saved.data?.length;
  return (
    <Section
      title="The saved models"
      topic="pickle_safety"
      className="mt-24 border-t border-border pt-14"
      actions={
        count !== undefined && (
          <p className="font-mono text-xs text-dim tabular-nums">
            {count === 1 ? "1 model" : `${String(count)} models`}
          </p>
        )
      }
    >
      <div className="flex flex-col gap-6">
        <LoadZone
          onLoaded={(pkg) => {
            // O pacote carregado abre logo no leitor; o de outro ML Lab, primeiro com o aviso.
            // Um de cada vez: dois modais abertos juntos escondiam-se um ao outro.
            if (!pkg.trusted && pkg.intact) {
              setWarning({ pkg, open: true, thenRead: true });
            } else {
              setOpened(pkg.package_id);
            }
          }}
        />
        <ErrorNote error={saved.error} />
        {saved.data === undefined ? (
          saved.error === null && <Skeleton className="h-40 w-full" />
        ) : saved.data.length === 0 ? (
          <p className="text-sm text-soft">
            No saved models yet: the ones you save, or load, are listed here.
          </p>
        ) : (
          <ul className="flex flex-col divide-y divide-border rounded-2xl border bg-surface">
            {saved.data.map((pkg) => (
              <li key={pkg.package_id} className="min-w-0">
                <PackageRow pkg={pkg} onOpen={setOpened} onTrust={ask} />
              </li>
            ))}
          </ul>
        )}
      </div>
      <PackageSheet
        packageId={opened}
        onTrust={ask}
        onClose={() => {
          setOpened(null);
        }}
      />
      <TrustDialog
        pkg={warning?.pkg ?? null}
        open={warning?.open ?? false}
        onClose={() => {
          setWarning((current) => current && { ...current, open: false });
        }}
        onClosed={() => {
          if (warning?.thenRead === true) {
            setOpened(warning.pkg.package_id);
          }
          setWarning(null);
        }}
      />
    </Section>
  );
}

/**
 * Um pacote da biblioteca, numa linha: a marca da cor da família do modelo, o nome com o modelo e
 * os dados, os números (a métrica, o tamanho, a data, a origem) e as ações.
 */
function PackageRow({
  pkg,
  onOpen,
  onTrust,
}: {
  pkg: SavedPackage;
  onOpen: (packageId: string) => void;
  onTrust: (pkg: SavedPackage) => void;
}) {
  const label = useLabel();
  const titleId = `pacote-${pkg.package_id}`;
  const color = pkg.family === null ? OTHER : familyColor(pkg.family);
  return (
    // Uma grelha: a marca à esquerda; o nome e, por baixo, o modelo e os dados; à direita, os
    // números ao nível do nome e as ações ao nível do modelo. Num ecrã estreito, tudo numa coluna.
    <div
      role="group"
      aria-labelledby={titleId}
      className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-2 px-5 py-4 @3xl/main:grid-cols-[auto_minmax(0,1fr)_auto] @3xl/main:gap-x-8"
    >
      <span
        aria-hidden="true"
        className="col-start-1 row-span-2 row-start-1 mt-2 size-2.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      <h3
        id={titleId}
        className="col-start-2 row-start-1 text-base font-medium tracking-normal break-words"
      >
        {pkg.name}
      </h3>
      <p className="col-start-2 row-start-2 text-sm text-soft">
        {[pkg.model_display_name, pkg.dataset_name].filter(Boolean).join(" · ")}
      </p>
      <div className="col-start-2 row-start-3 flex flex-col gap-3 @3xl/main:col-start-3 @3xl/main:row-span-2 @3xl/main:row-start-1 @3xl/main:items-end @3xl/main:gap-2">
        <dl className="flex flex-wrap items-center gap-x-5 gap-y-1 font-mono text-xs text-dim tabular-nums">
          {pkg.primary_label !== null && pkg.primary_value !== null && (
            <div className="flex gap-1.5">
              <dt>{pkg.primary_label}</dt>
              <dd className="text-strong">{metricText(pkg)}</dd>
            </div>
          )}
          <div>
            <dt className="sr-only">Size</dt>
            <dd>{formatBytes(pkg.size)}</dd>
          </div>
          <div>
            <dt className="sr-only">In the library since</dt>
            <dd>{formatMoment(pkg.added_at)}</dd>
          </div>
          <div>
            <dt className="sr-only">Origin</dt>
            <dd className="rounded-full border border-[#333] px-2.5 py-0.5 font-sans font-medium text-soft">
              {label("PackageOrigin", pkg.origin)}
            </dd>
          </div>
        </dl>
        <div className="flex flex-wrap items-center gap-1.5">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              onOpen(pkg.package_id);
            }}
          >
            Open
          </Button>
          {!pkg.trusted && pkg.intact && (
            <Button
              size="sm"
              onClick={() => {
                onTrust(pkg);
              }}
            >
              Open the model
            </Button>
          )}
          {pkg.trusted && pkg.run_id !== null && (
            <Button asChild variant="ghost" size="sm">
              <Link to="/train/$runId" params={{ runId: pkg.run_id }}>
                Train & evaluate
              </Link>
            </Button>
          )}
          <Button asChild variant="ghost" size="sm">
            <a
              href={packageFileUrl(pkg.package_id)}
              download={pkg.file_name}
              aria-label={`Download ${pkg.name}`}
            >
              <Download aria-hidden="true" />
            </a>
          </Button>
          <DeleteButton pkg={pkg} />
        </div>
      </div>
      {pkg.notices.length > 0 && (
        <div className="col-start-2 row-start-4 @3xl/main:col-span-2 @3xl/main:row-start-3">
          <NoticeList notices={pkg.notices} />
        </div>
      )}
    </div>
  );
}

function metricText(pkg: SavedPackage): string {
  if (pkg.primary_value === null) {
    return "—";
  }
  return pkg.primary_as_percent
    ? formatPercent(pkg.primary_value)
    : formatNumber(pkg.primary_value, 3);
}

/** Apagar um pacote da biblioteca, depois de confirmar. */
function DeleteButton({ pkg }: { pkg: SavedPackage }) {
  const queryClient = useQueryClient();
  const remove = useMutation({
    ...packagesDeleteMutation(),
    onSuccess: () => {
      invalidatePackages(queryClient);
    },
  });
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="ghost" size="sm" aria-label={`Delete ${pkg.name}`}>
          <Trash2 aria-hidden="true" />
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete “{pkg.name}”?</AlertDialogTitle>
          <AlertDialogDescription>
            The package leaves the library. The run it put in the history stays there.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              remove.mutate({ path: { package_id: pkg.package_id } });
            }}
          >
            Delete
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

/** O aviso antes de abrir o modelo de um pacote feito noutro ML Lab. */
function TrustDialog({
  pkg,
  open,
  onClose,
  onClosed,
}: {
  pkg: SavedPackage | null;
  open: boolean;
  onClose: () => void;
  /** Depois de fechado de vez (no fim da animação): o leitor pode abrir sem se sobrepor. */
  onClosed: () => void;
}) {
  const queryClient = useQueryClient();
  const trust = useMutation({
    ...packagesTrustMutation(),
    onSuccess: (trusted) => {
      invalidatePackages(queryClient);
      toast.success(`The model in “${trusted.name}” is open: the run is in the history.`);
    },
    onError: (error) => {
      toast.error(error.message);
    },
  });
  const warnings = pkg?.notices ?? [];
  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        if (!next) {
          onClose();
        }
      }}
    >
      <AlertDialogContent
        className="sm:max-w-lg"
        onCloseAutoFocus={() => {
          // O foco volta ao botão que abriu o aviso; se o leitor abrir, passa para ele.
          onClosed();
        }}
      >
        <AlertDialogHeader>
          <AlertDialogTitle>Open a model made in another Orbyte?</AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="flex flex-col gap-2 text-sm leading-relaxed text-soft">
              {warnings.map((notice) => (
                <p key={notice.code}>{notice.message}</p>
              ))}
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Just read it</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              if (pkg !== null) {
                trust.mutate({ path: { package_id: pkg.package_id } });
              }
            }}
          >
            I trust it, open the model
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

/** Um pacote da biblioteca aberto ao lado: o que se sabe dele e as suas partes. */
function PackageSheet({
  packageId,
  onTrust,
  onClose,
}: {
  packageId: string | null;
  onTrust: (pkg: SavedPackage) => void;
  onClose: () => void;
}) {
  return (
    <Sheet
      open={packageId !== null}
      onOpenChange={(open) => {
        if (!open) {
          onClose();
        }
      }}
    >
      <SheetContent
        side="right"
        className="w-full gap-0 overflow-y-auto data-[side=right]:sm:max-w-4xl"
      >
        {packageId !== null && <PackageDetails packageId={packageId} onTrust={onTrust} />}
      </SheetContent>
    </Sheet>
  );
}

function PackageDetails({
  packageId,
  onTrust,
}: {
  packageId: string;
  onTrust: (pkg: SavedPackage) => void;
}) {
  const label = useLabel();
  const view = useQuery(packagesContentsOptions({ path: { package_id: packageId } }));
  if (view.data === undefined) {
    return (
      <div className="flex flex-col gap-4 p-6">
        <SheetHeader className="p-0">
          <SheetTitle>The package</SheetTitle>
          <SheetDescription className="sr-only">Loading the package.</SheetDescription>
        </SheetHeader>
        <ErrorNote error={view.error} />
        {view.error === null && <Skeleton className="h-96 w-full" />}
      </div>
    );
  }
  const { package: pkg, preview } = view.data;
  return (
    <div className="flex flex-col gap-6 p-6">
      <SheetHeader className="gap-1 p-0">
        <SheetTitle className="text-xl font-medium text-strong">{pkg.name}</SheetTitle>
        <SheetDescription className="font-mono text-xs text-dim">{pkg.file_name}</SheetDescription>
      </SheetHeader>
      <div className="flex flex-wrap items-center gap-2">
        {pkg.trusted && pkg.run_id !== null && (
          <Button asChild variant="outline" size="sm">
            <Link to="/train/$runId" params={{ runId: pkg.run_id }}>
              Open in Train & evaluate
            </Link>
          </Button>
        )}
        {!pkg.trusted && pkg.intact && (
          <Button
            size="sm"
            onClick={() => {
              onTrust(pkg);
            }}
          >
            Open the model
          </Button>
        )}
        <Button asChild variant="ghost" size="sm">
          <a href={packageFileUrl(pkg.package_id)} download={pkg.file_name}>
            <Download aria-hidden="true" />
            Download
          </a>
        </Button>
      </div>
      <StatStrip
        compact
        items={[
          {
            key: "model",
            label: "Model",
            value: <span className="text-lg">{pkg.model_display_name}</span>,
          },
          {
            key: "data",
            label: "Data",
            value: <span className="text-lg">{pkg.dataset_name}</span>,
          },
          {
            key: "origin",
            label: "Origin",
            value: <span className="text-lg">{label("PackageOrigin", pkg.origin)}</span>,
          },
          {
            key: "size",
            label: "Size",
            value: <span className="text-lg">{formatBytes(pkg.size)}</span>,
          },
        ]}
      />
      {pkg.notices.length > 0 && <NoticeList notices={pkg.notices} />}
      <PackageViewer parts={preview.parts} label="The package's parts" />
    </div>
  );
}

/** Carregar um modelo treinado: larga-se um `.mllab`, ou clica-se para o escolher. */
function LoadZone({ onLoaded }: { onLoaded: (pkg: SavedPackage) => void }) {
  const queryClient = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const load = useMutation({
    ...packagesLoadMutation(),
    onSuccess: (pkg) => {
      invalidatePackages(queryClient);
      setFile(null);
      if (pkg.trusted) {
        toast.success(`“${pkg.name}” is in the library, and its run in the history.`);
      } else if (!pkg.intact) {
        toast.warning(`“${pkg.name}” is damaged: it stays in the library, read-only.`);
      }
      onLoaded(pkg);
    },
  });

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files.item(0);
    if (dropped !== null) {
      setFile(dropped);
      load.reset();
    }
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Uma faixa, e não uma caixa alta: carregar é a ação secundária da biblioteca. */}
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => {
          setDragging(false);
        }}
        onDrop={drop}
        className={cn(
          "relative flex min-h-20 flex-wrap items-center gap-x-4 gap-y-3 rounded-2xl border border-dashed px-5 py-4 transition-colors",
          "focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50",
          dragging
            ? "border-strong/70 bg-surface-high"
            : "border-[#3a3a3a] bg-ground hover:border-[#5a5a5a] hover:bg-[#151515]",
        )}
      >
        {/* A zona toda escolhe o ficheiro; só o botão de carregar fica por cima dela. */}
        <label htmlFor="pacote-ficheiro" className="absolute inset-0 cursor-pointer rounded-2xl">
          <span className="sr-only">Choose a .mllab file from your computer</span>
        </label>
        <span
          aria-hidden="true"
          className={cn(
            "pointer-events-none flex size-11 shrink-0 items-center justify-center rounded-xl border bg-surface transition-transform duration-300 motion-reduce:transition-none",
            dragging ? "scale-110 border-[#5a5a5a]" : "border-[#333]",
          )}
        >
          {file === null ? (
            <FileUp className="size-5 text-soft" />
          ) : (
            <FileArchive className="size-5 text-strong" />
          )}
        </span>
        <div className="pointer-events-none flex min-w-0 flex-1 flex-col gap-0.5">
          <p className="text-base font-medium text-strong">Load a trained model</p>
          <p className="text-sm break-words text-soft">
            {file === null
              ? dragging
                ? "Drop the file here"
                : "Drop a .mllab file here, or click to choose one"
              : `${file.name} · ${formatBytes(file.size)}`}
          </p>
        </div>
        {file !== null && (
          <Button
            className="relative z-10"
            disabled={load.isPending}
            onClick={() => {
              load.mutate({ body: { file } });
            }}
          >
            {load.isPending && <LoaderCircle aria-hidden="true" className="animate-spin" />}
            Load the model
          </Button>
        )}
        <input
          id="pacote-ficheiro"
          type="file"
          accept=".mllab,application/zip"
          className="sr-only"
          onChange={(event) => {
            setFile(event.target.files?.item(0) ?? null);
            load.reset();
          }}
        />
      </div>
      <ErrorNote error={load.error} />
    </div>
  );
}
