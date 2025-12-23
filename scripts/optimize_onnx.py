# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "onnx",
#     "onnxruntime",
#     "onnxoptimizer",
#     "rich",
#     "typer",
# ]
# ///
"""
ONNX model optimizer script.

Applies various optimization passes to reduce model complexity:
- Shape inference to resolve dynamic dimensions
- ONNX optimizer passes (constant folding, shape elimination, etc.)
- ONNX Runtime graph optimization (node fusion, redundant node elimination)
- Optional: fix input shapes to enable more aggressive optimization
"""

import typer
from pathlib import Path
from typing import Optional, Annotated
from rich.console import Console
from rich.table import Table

import onnx
from onnx import shape_inference, numpy_helper
import onnxoptimizer
import onnxruntime as ort

console = Console()
app = typer.Typer(
    help="Optimize ONNX models by applying various optimization passes",
    add_completion=False,
    no_args_is_help=True,
)


def get_dynamic_dims(model: onnx.ModelProto) -> set[str]:
    """Extract all dynamic dimension parameter names from the model."""
    dim_params = set()

    def extract_dims(type_proto):
        if type_proto.HasField("tensor_type"):
            for d in type_proto.tensor_type.shape.dim:
                if d.dim_param:
                    dim_params.add(d.dim_param)

    for i in model.graph.input:
        extract_dims(i.type)
    for o in model.graph.output:
        extract_dims(o.type)
    for vi in model.graph.value_info:
        extract_dims(vi.type)

    return dim_params


def get_model_stats(model: onnx.ModelProto) -> dict:
    """Get statistics about the model."""
    from collections import Counter

    node_types = Counter(n.op_type for n in model.graph.node)

    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "inputs": len(
            [
                i
                for i in model.graph.input
                if not any(init.name == i.name for init in model.graph.initializer)
            ]
        ),
        "outputs": len(model.graph.output),
        "dynamic_dims": len(get_dynamic_dims(model)),
        "Shape ops": node_types.get("Shape", 0),
        "Constant ops": node_types.get("Constant", 0),
        "Unsqueeze ops": node_types.get("Unsqueeze", 0),
        "Concat ops": node_types.get("Concat", 0),
    }


def display_comparison(before: dict, after: dict, title: str = "Optimization Results"):
    """Display before/after comparison table."""
    table = Table(title=title)
    table.add_column("Metric", style="bold blue")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Change", justify="right")

    for key in before:
        b, a = before[key], after[key]
        diff = a - b
        if diff < 0:
            change = f"[green]{diff}[/green]"
        elif diff > 0:
            change = f"[red]+{diff}[/red]"
        else:
            change = "—"

        table.add_row(
            key,
            str(b),
            str(a),
            change,
        )

    console.print(table)


def apply_shape_inference(model: onnx.ModelProto) -> onnx.ModelProto:
    """Apply shape inference to resolve as many shapes as possible."""
    return shape_inference.infer_shapes(model, data_prop=True)


def fix_input_shapes(
    model: onnx.ModelProto,
    input_shapes: dict[str, list[int]],
) -> onnx.ModelProto:
    """Fix specific input dimensions to enable static shape optimization."""
    graph = model.graph

    for inp in graph.input:
        if inp.name in input_shapes:
            shape = input_shapes[inp.name]
            tensor_type = inp.type.tensor_type

            # Clear existing dims and set new ones
            while len(tensor_type.shape.dim) > 0:
                tensor_type.shape.dim.pop()

            for dim_val in shape:
                dim = tensor_type.shape.dim.add()
                if isinstance(dim_val, str):
                    dim.dim_param = dim_val
                else:
                    dim.dim_value = dim_val

    return model


def apply_onnx_optimizer(
    model: onnx.ModelProto, passes: list[str] | None = None
) -> onnx.ModelProto:
    """Apply ONNX optimizer passes."""
    if passes is None:
        # Use passes that help with shape/constant folding
        passes = [
            # Shape-related optimizations
            "eliminate_shape_gather",
            "eliminate_slice_after_shape",
            "eliminate_shape_op",
            # Constant/identity elimination
            "extract_constant_to_initializer",
            "eliminate_identity",
            "eliminate_deadend",
            "eliminate_unused_initializer",
            "eliminate_duplicate_initializer",
            # NOP elimination
            "eliminate_nop_cast",
            "eliminate_nop_dropout",
            "eliminate_nop_flatten",
            "eliminate_nop_monotone_argmax",
            "eliminate_nop_pad",
            "eliminate_nop_concat",
            "eliminate_nop_split",
            "eliminate_nop_expand",
            "eliminate_nop_transpose",
            "eliminate_nop_reshape",
            "eliminate_nop_with_unit",
            # Common subexpression elimination
            "eliminate_common_subexpression",
            # Fusion passes
            "fuse_consecutive_concats",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_consecutive_unsqueezes",
            "fuse_consecutive_slices",
            "fuse_concat_into_reshape",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_pad_into_conv",
            "fuse_pad_into_pool",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_transpose_into_gemm",
        ]

    return onnxoptimizer.optimize(model, passes)


def optimize_with_onnxruntime(
    model_path: Path,
    output_path: Path,
    opt_level: ort.GraphOptimizationLevel,
) -> None:
    """Use ONNX Runtime's graph optimizers."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = opt_level
    sess_options.optimized_model_filepath = str(output_path)

    # Create session to trigger optimization and save
    ort.InferenceSession(
        str(model_path),
        sess_options,
        providers=["CPUExecutionProvider"],
    )


def fold_constants(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Fold constant expressions in the graph.

    This helps reduce patterns like Shape -> Gather -> Unsqueeze -> Concat
    when the shapes are actually known.
    """
    # First run shape inference to populate value_info
    model = apply_shape_inference(model)

    graph = model.graph

    # Build a map of constant values
    constants = {}

    # Get constants from initializers
    for init in graph.initializer:
        constants[init.name] = numpy_helper.to_array(init)

    # Get constants from Constant nodes
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)

    # We could implement more sophisticated constant folding here,
    # but the onnxoptimizer passes + ORT optimization handle most cases
    return model


def parse_input_shapes(shape_strs: list[str]) -> dict[str, list[int | str]]:
    """Parse input shape specifications like 'input_name:1,3,224,224'."""
    shapes = {}
    for spec in shape_strs:
        if ":" not in spec:
            raise ValueError(
                f"Invalid shape spec '{spec}'. Use format 'name:dim1,dim2,...'"
            )

        name, dims_str = spec.split(":", 1)
        dims = []
        for d in dims_str.split(","):
            d = d.strip()
            try:
                dims.append(int(d))
            except ValueError:
                # Keep as string (dynamic dim name)
                dims.append(d)
        shapes[name] = dims

    return shapes


@app.command()
def optimize(
    input_file: Annotated[Path, typer.Argument(help="Input ONNX model file")],
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output ONNX file (default: input_optimized.onnx)",
        ),
    ] = None,
    input_shape: Annotated[
        list[str],
        typer.Option(
            "--input-shape",
            "-s",
            help="Fix input shape, e.g., 'pixel_values:1,3,640,640'. Can be specified multiple times.",
        ),
    ] = [],
    opt_level: Annotated[
        str,
        typer.Option(
            "--opt-level",
            "-l",
            help="ONNX Runtime optimization level: disabled, basic, extended, all",
        ),
    ] = "basic",
    use_ort: Annotated[
        bool,
        typer.Option(
            "--use-ort/--skip-ort",
            help="Enable/skip ONNX Runtime optimization (disabled by default, as ORT can introduce new dynamic dims)",
        ),
    ] = False,
    skip_onnx_opt: Annotated[
        bool,
        typer.Option(
            "--skip-onnx-opt",
            help="Skip ONNX optimizer passes",
        ),
    ] = False,
    infer_shapes: Annotated[
        bool,
        typer.Option(
            "--infer-shapes/--no-infer-shapes",
            help="Run ONNX shape inference pass",
        ),
    ] = True,
    iterations: Annotated[
        int,
        typer.Option(
            "--iterations",
            "-i",
            help="Number of optimization iterations (more iterations may catch cascading simplifications)",
        ),
    ] = 3,
):
    """
    Optimize an ONNX model.

    Applies multiple optimization passes:
    1. Shape inference to resolve tensor shapes
    2. ONNX optimizer passes (shape elimination, constant folding, fusion)
    3. ONNX Runtime graph optimization

    Examples:

        # Basic optimization with all passes
        uv run scripts/optimize_onnx.py model.onnx

        # Fix batch size to 1 for more aggressive optimization
        uv run scripts/optimize_onnx.py model.onnx -s "pixel_values:1,3,640,640" -s "pixel_mask:1,64,64"

        # Only run ONNX optimizer (skip ORT)
        uv run scripts/optimize_onnx.py model.onnx --skip-ort
    """
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)

    if output_file is None:
        output_file = input_file.with_stem(f"{input_file.stem}_optimized")

    # Map string to optimization level
    opt_levels = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    if opt_level not in opt_levels:
        console.print(f"[red]Error: Invalid optimization level '{opt_level}'[/red]")
        console.print(f"Valid options: {', '.join(opt_levels.keys())}")
        raise typer.Exit(1)

    try:
        console.print(f"[blue]Loading model from {input_file}...[/blue]")
        model = onnx.load(str(input_file))
        before_stats = get_model_stats(model)

        console.print(
            f"[dim]  Nodes: {before_stats['nodes']}, Dynamic dims: {before_stats['dynamic_dims']}[/dim]"
        )

        # Fix input shapes if specified
        if input_shape:
            console.print(f"[blue]Fixing input shapes: {input_shape}[/blue]")
            shapes = parse_input_shapes(input_shape)
            model = fix_input_shapes(model, shapes)

        # Run optimization iterations
        for iteration in range(iterations):
            prev_nodes = len(model.graph.node)

            # Run shape inference
            if infer_shapes:
                if iteration == 0:
                    console.print("[blue]Running shape inference...[/blue]")
                model = apply_shape_inference(model)

            # Run ONNX optimizer passes
            if not skip_onnx_opt:
                if iteration == 0:
                    console.print("[blue]Running ONNX optimizer passes...[/blue]")
                model = apply_onnx_optimizer(model)

            # Run shape inference again after optimizer
            if infer_shapes:
                model = apply_shape_inference(model)

            curr_nodes = len(model.graph.node)
            if curr_nodes == prev_nodes and iteration > 0:
                console.print(
                    f"[dim]  Converged after {iteration + 1} iterations[/dim]"
                )
                break

            if iteration > 0:
                console.print(
                    f"[dim]  Iteration {iteration + 1}: {prev_nodes} → {curr_nodes} nodes[/dim]"
                )

        # Run ONNX Runtime optimization if requested
        if use_ort:
            # Save intermediate model for ORT optimization
            temp_path = output_file.with_stem(f"{output_file.stem}_temp")
            onnx.save(model, str(temp_path))

            try:
                console.print(
                    f"[blue]Running ONNX Runtime optimization (level: {opt_level})...[/blue]"
                )
                optimize_with_onnxruntime(temp_path, output_file, opt_levels[opt_level])
                temp_path.unlink()  # Clean up temp file

                # Reload optimized model for stats
                model = onnx.load(str(output_file))
            except Exception as e:
                temp_path.unlink(missing_ok=True)
                console.print(
                    f"[yellow]Warning: ORT optimization failed ({e}), using ONNX optimizer output only[/yellow]"
                )
                onnx.save(model, str(output_file))
        else:
            onnx.save(model, str(output_file))

        # Run shape inference one more time on the output for clean shapes
        if infer_shapes:
            model = apply_shape_inference(model)
            onnx.save(model, str(output_file))

        after_stats = get_model_stats(model)

        console.print()
        display_comparison(before_stats, after_stats)

        # Show file size comparison
        input_size = input_file.stat().st_size
        output_size = output_file.stat().st_size
        size_diff = output_size - input_size
        size_pct = (size_diff / input_size) * 100

        console.print()
        console.print(f"[dim]Input size:  {input_size / 1024 / 1024:.1f} MB[/dim]")
        console.print(
            f"[dim]Output size: {output_size / 1024 / 1024:.1f} MB ({size_pct:+.1f}%)[/dim]"
        )

        # Show remaining dynamic dims
        remaining_dims = get_dynamic_dims(model)
        if remaining_dims:
            # Filter to show only meaningful ones (not unk__)
            meaningful = [d for d in remaining_dims if not d.startswith("unk__")]
            unknown = [d for d in remaining_dims if d.startswith("unk__")]

            if meaningful:
                console.print(
                    f"\n[yellow]Remaining dynamic dimensions: {', '.join(sorted(meaningful))}[/yellow]"
                )
            if unknown:
                console.print(f"[dim]  + {len(unknown)} unk__ dimensions[/dim]")

        console.print(f"\n[green]✓ Optimized model saved to {output_file}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def info(
    input_file: Annotated[Path, typer.Argument(help="ONNX model file to inspect")],
    show_dims: Annotated[
        bool,
        typer.Option("--dims", "-d", help="Show all dynamic dimension names"),
    ] = False,
    show_ops: Annotated[
        bool,
        typer.Option("--ops", help="Show operation type counts"),
    ] = False,
):
    """Display information about an ONNX model."""
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)

    model = onnx.load(str(input_file))

    console.print(f"[bold cyan]{input_file.name}[/bold cyan]")
    console.print(f"  IR version: {model.ir_version}")
    opsets = [f"{x.domain or 'ai.onnx'}:{x.version}" for x in model.opset_import]
    console.print(f"  Opset: {opsets}")
    console.print(f"  Nodes: {len(model.graph.node)}")
    console.print(f"  Initializers: {len(model.graph.initializer)}")

    console.print("\n[bold]Inputs:[/bold]")
    for inp in model.graph.input:
        # Skip initializers that appear as inputs
        if any(init.name == inp.name for init in model.graph.initializer):
            continue
        dims = []
        if inp.type.tensor_type.shape.dim:
            for d in inp.type.tensor_type.shape.dim:
                if d.dim_param:
                    dims.append(f"[yellow]{d.dim_param}[/yellow]")
                else:
                    dims.append(str(d.dim_value))
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        console.print(f"  {inp.name}: [{' × '.join(dims)}] ({dtype})")

    console.print("\n[bold]Outputs:[/bold]")
    for out in model.graph.output:
        dims = []
        if out.type.tensor_type.shape.dim:
            for d in out.type.tensor_type.shape.dim:
                if d.dim_param:
                    dims.append(f"[yellow]{d.dim_param}[/yellow]")
                else:
                    dims.append(str(d.dim_value))
        dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        console.print(f"  {out.name}: [{' × '.join(dims)}] ({dtype})")

    dynamic_dims = get_dynamic_dims(model)
    console.print(f"\n[bold]Dynamic dimensions:[/bold] {len(dynamic_dims)}")

    if show_dims and dynamic_dims:
        meaningful = sorted(d for d in dynamic_dims if not d.startswith("unk__"))
        unknown = sorted(d for d in dynamic_dims if d.startswith("unk__"))

        if meaningful:
            console.print(f"  Named: {', '.join(meaningful)}")
        if unknown:
            console.print(
                f"  Unknown: {len(unknown)} (unk__0 through unk__{len(unknown) - 1})"
            )

    if show_ops:
        from collections import Counter

        node_types = Counter(n.op_type for n in model.graph.node)

        console.print("\n[bold]Operation counts:[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Op Type")
        table.add_column("Count", justify="right")

        for op, count in node_types.most_common():
            table.add_row(op, str(count))

        console.print(table)


@app.command()
def passes():
    """List all available ONNX optimizer passes."""
    console.print("[bold]Available ONNX optimizer passes:[/bold]\n")

    all_passes = onnxoptimizer.get_available_passes()

    # Group by category
    categories = {
        "eliminate": [],
        "fuse": [],
        "other": [],
    }

    for p in all_passes:
        if p.startswith("eliminate"):
            categories["eliminate"].append(p)
        elif p.startswith("fuse"):
            categories["fuse"].append(p)
        else:
            categories["other"].append(p)

    for cat, cat_passes in categories.items():
        console.print(f"[bold blue]{cat.title()}:[/bold blue]")
        for p in sorted(cat_passes):
            console.print(f"  - {p}")
        console.print()


if __name__ == "__main__":
    app()
