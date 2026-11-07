"""Execute and save the submission notebook; full main experiments are never launched."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=ROOT/'notebooks/LipoQ_Submission.ipynb')
    parser.add_argument('--timeout', type=int, default=180, help='Per-cell timeout in seconds')
    parser.add_argument('--working-directory', type=Path, default=ROOT,
                        help='Kernel working directory; use an empty directory to check portability')
    args = parser.parse_args()
    args.notebook = args.notebook.resolve()
    runtime = ROOT/'.local/notebook_execution'
    for variable, directory in [('IPYTHONDIR','ipython'), ('JUPYTER_RUNTIME_DIR','runtime'), ('JUPYTER_CONFIG_DIR','config')]:
        target = runtime/directory
        target.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(target))
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager
    kernel_dir = runtime/'kernels/lipoq'
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir/'kernel.json').write_text(json.dumps({
        'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Python (LipoQ)', 'language':'python'}))
    notebook = nbformat.read(args.notebook, as_version=4)
    if notebook.metadata.get('lipoq_single_file'):
        # The standalone notebook also works without this helper. Reuse local
        # verified assets when executing from the repository to avoid downloads.
        for variable, asset in [('LIPOQ_COMET_REPO', ROOT/'vendor/COMET'),
                                ('LIPOQ_COMET_CHECKPOINT', ROOT/'artifacts/comet_lipid_fold0.pt')]:
            if asset.exists():
                os.environ.setdefault(variable, str(asset))
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    nbformat.validate(notebook)
    started = time.perf_counter()
    notebook_label = args.notebook.relative_to(ROOT) if args.notebook.is_relative_to(ROOT) else args.notebook
    report = {'status':'running','notebook':str(notebook_label)}

    def completed(cell, cell_index, **kwargs):
        print(f'Executed cell {cell_index+1}/{len(notebook.cells)}', flush=True)

    client = NotebookClient(notebook, kernel_name='lipoq', timeout=args.timeout,
                            resources={'metadata':{'path':str(args.working_directory.resolve())}},
                            allow_errors=False, on_cell_executed=completed)
    manager = client.create_kernel_manager()
    manager.kernel_spec_manager = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
    try:
        client.execute()
        errors = [output for cell in notebook.cells if cell.cell_type=='code'
                  for output in cell.outputs if output.output_type=='error']
        if errors:
            raise RuntimeError('Notebook contains error outputs')
        report.update(status='completed', code_cells=sum(c.cell_type=='code' for c in notebook.cells),
                      embedded_images=sum('image/png' in o.get('data',{}) for c in notebook.cells for o in c.get('outputs',[])))
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['seconds'] = time.perf_counter()-started
        notebook.metadata['lipoq_execution'] = report
        if notebook.metadata.get('lipoq_single_file'):
            notebook.metadata['kernelspec'] = {'name':'python3', 'language':'python',
                                              'display_name':'Python 3 (ipykernel)'}
        nbformat.validate(notebook)
        temp = args.notebook.with_suffix('.tmp.ipynb')
        nbformat.write(notebook, temp)
        temp.replace(args.notebook)
        (runtime/f'{args.notebook.stem}.report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
