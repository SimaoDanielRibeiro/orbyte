/**
 * A biblioteca dos modelos guardados, e o carregar de um modelo já treinado (ADR 0039).
 *
 * Cada pacote é um cartão com o contorno da cor da família do modelo (ADR 0038): o nome, o
 * modelo e os dados, a métrica principal e de onde veio. Abre-se ao lado, no mesmo leitor da
 * pré-visualização, descarrega-se ou apaga-se.
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
import { withAlpha } from "@/lib/colors";
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
  return (
    <Section title="Os modelos guardados" topic="pickle_safety" className="mt-24">
      <div className="grid grid-cols-1 gap-x-12 gap-y-10 @5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,24rem)]">
        <div className="min-w-0">
          <ErrorNote error={saved.error} />
          {saved.data === undefined ? (
            saved.error === null && <Skeleton className="h-40 w-full" />
          ) : saved.data.length === 0 ? (
            <p className="text-sm text-soft">Ainda não há nenhum modelo guardado.</p>
          ) : (
            <ul className="grid grid-cols-1 gap-4 @6xl/main:grid-cols-2">
              {saved.data.map((pkg) => (
                <li key={pkg.package_id} className="min-w-0">
                  <PackageCard pkg={pkg} onOpen={setOpened} onTrust={ask} />
                </li>
              ))}
            </ul>
          )}
        </div>
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

/** Um pacote da biblioteca, num cartão com o contorno da cor da família do modelo. */
function PackageCard({
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
  const color = pkg.family === null ? null : familyColor(pkg.family);
  return (
    <div
      role="group"
      aria-labelledby={titleId}
      className="flex h-full flex-col gap-3 rounded-2xl border bg-surface p-4"
      style={color === null ? undefined : { borderColor: withAlpha(color, 0.45) }}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 id={titleId} className="text-[0.9375rem] font-medium tracking-normal break-words">
            {pkg.name}
          </h3>
          <p className="mt-1 text-sm text-soft">
            {[pkg.model_display_name, pkg.dataset_name].filter(Boolean).join(" · ")}
          </p>
        </div>
        <span className="shrink-0 rounded-full border border-[#333] px-2.5 py-0.5 font-mono text-[0.6875rem] text-soft">
          {label("PackageOrigin", pkg.origin)}
        </span>
      </div>
      <dl className="flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-dim">
        {pkg.primary_label !== null && pkg.primary_value !== null && (
          <div className="flex gap-1.5">
            <dt>{pkg.primary_label}</dt>
            <dd className="text-strong tabular-nums">{metricText(pkg)}</dd>
          </div>
        )}
        <div className="flex gap-1.5">
          <dt className="sr-only">Tamanho</dt>
          <dd className="tabular-nums">{formatBytes(pkg.size)}</dd>
        </div>
        <div className="flex gap-1.5">
          <dt className="sr-only">Na biblioteca desde</dt>
          <dd className="tabular-nums">{formatMoment(pkg.added_at)}</dd>
        </div>
      </dl>
      {pkg.notices.length > 0 && <NoticeList notices={pkg.notices} />}
      <div className="mt-auto flex flex-wrap items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            onOpen(pkg.package_id);
          }}
        >
          Abrir
        </Button>
        {!pkg.trusted && pkg.intact && (
          <Button
            size="sm"
            onClick={() => {
              onTrust(pkg);
            }}
          >
            Abrir o modelo
          </Button>
        )}
        {pkg.trusted && pkg.run_id !== null && (
          <Button asChild variant="outline" size="sm">
            <Link to="/treinar/$runId" params={{ runId: pkg.run_id }}>
              Abrir no Treinar e avaliar
            </Link>
          </Button>
        )}
        <Button asChild variant="ghost" size="sm">
          <a href={packageFileUrl(pkg.package_id)} download={pkg.file_name}>
            <Download aria-hidden="true" />
            Descarregar
          </a>
        </Button>
        <DeleteButton pkg={pkg} />
      </div>
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
        <Button variant="ghost" size="sm" aria-label={`Apagar ${pkg.name}`}>
          <Trash2 aria-hidden="true" />
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Apagar «{pkg.name}»?</AlertDialogTitle>
          <AlertDialogDescription>
            O pacote sai da biblioteca. O run que ele pôs no histórico fica lá.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancelar</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              remove.mutate({ path: { package_id: pkg.package_id } });
            }}
          >
            Apagar
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
      toast.success(`O modelo de «${trusted.name}» abriu: o run está no histórico.`);
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
          <AlertDialogTitle>Abrir um modelo feito noutro ML Lab?</AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="flex flex-col gap-2 text-sm leading-relaxed text-soft">
              {warnings.map((notice) => (
                <p key={notice.code}>{notice.message}</p>
              ))}
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Só ler</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              if (pkg !== null) {
                trust.mutate({ path: { package_id: pkg.package_id } });
              }
            }}
          >
            Confio, abrir o modelo
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
          <SheetTitle>O pacote</SheetTitle>
          <SheetDescription className="sr-only">A carregar o pacote.</SheetDescription>
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
            <Link to="/treinar/$runId" params={{ runId: pkg.run_id }}>
              Abrir no Treinar e avaliar
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
            Abrir o modelo
          </Button>
        )}
        <Button asChild variant="ghost" size="sm">
          <a href={packageFileUrl(pkg.package_id)} download={pkg.file_name}>
            <Download aria-hidden="true" />
            Descarregar
          </a>
        </Button>
      </div>
      <StatStrip
        compact
        items={[
          {
            key: "modelo",
            label: "Modelo",
            value: <span className="text-lg">{pkg.model_display_name}</span>,
          },
          {
            key: "dados",
            label: "Dados",
            value: <span className="text-lg">{pkg.dataset_name}</span>,
          },
          {
            key: "origem",
            label: "Origem",
            value: <span className="text-lg">{label("PackageOrigin", pkg.origin)}</span>,
          },
          {
            key: "tamanho",
            label: "Tamanho",
            value: <span className="text-lg">{formatBytes(pkg.size)}</span>,
          },
        ]}
      />
      {pkg.notices.length > 0 && <NoticeList notices={pkg.notices} />}
      <PackageViewer parts={preview.parts} label="As partes do pacote" />
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
        toast.success(`«${pkg.name}» está na biblioteca, e o run no histórico.`);
      } else if (!pkg.intact) {
        toast.warning(`«${pkg.name}» está estragado: fica na biblioteca só para ler.`);
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
      <h3 className="text-[0.9375rem] font-medium tracking-normal">Carregar um modelo treinado</h3>
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
          "relative flex min-h-56 flex-col items-center justify-center gap-5 rounded-2xl border border-dashed px-6 py-10 text-center transition-colors",
          "focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50",
          dragging
            ? "border-strong/70 bg-surface-high"
            : "border-[#3a3a3a] bg-ground hover:border-[#5a5a5a] hover:bg-[#151515]",
        )}
      >
        {/* A zona toda escolhe o ficheiro; só o botão de carregar fica por cima dela. */}
        <label htmlFor="pacote-ficheiro" className="absolute inset-0 cursor-pointer rounded-2xl">
          <span className="sr-only">Escolher um ficheiro .mllab do computador</span>
        </label>
        <span
          aria-hidden="true"
          className={cn(
            "pointer-events-none flex size-14 items-center justify-center rounded-2xl border bg-surface transition-transform duration-300 motion-reduce:transition-none",
            dragging ? "scale-110 border-[#5a5a5a]" : "border-[#333]",
          )}
        >
          {file === null ? (
            <FileUp className="size-6 text-soft" />
          ) : (
            <FileArchive className="size-6 text-strong" />
          )}
        </span>
        <p className="pointer-events-none text-[0.9375rem] text-strong">
          {file === null
            ? dragging
              ? "Larga o ficheiro aqui"
              : "Larga aqui um ficheiro .mllab, ou clica para o escolher"
            : `${file.name} · ${formatBytes(file.size)}`}
        </p>
        {file !== null && (
          <Button
            className="relative z-10"
            disabled={load.isPending}
            onClick={() => {
              load.mutate({ body: { file } });
            }}
          >
            {load.isPending && <LoaderCircle aria-hidden="true" className="animate-spin" />}
            Carregar o modelo
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
