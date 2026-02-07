import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ast
from collections import Counter, defaultdict

# Page config
st.set_page_config(page_title="NodeSynth Taxonomy", page_icon="🔗", layout="wide")


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = read_csv(path)
    return df


def normalize_cell(v):
    # Handle list-like strings and NaNs
    if pd.isna(v):
        return ""
    if isinstance(v, list):
        return ", ".join([str(x).strip() for x in v if x is not None])
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return ", ".join([str(x).strip() for x in parsed if x is not None])
            except Exception:
                pass
        return s
    return str(v)


def build_sankey(df: pd.DataFrame, dims: list, max_per_dim: int = 50, min_link_value: int = 1):
    # Collect most frequent values per dimension
    dim_values = {}
    for d in dims:
        vals = df[d].dropna().map(normalize_cell)
        # flatten comma-joined lists into individual items
        flat = []
        for v in vals:
            if "," in v:
                parts = [p.strip() for p in v.split(",") if p.strip()]
                flat.extend(parts)
            elif v:
                flat.append(v)
        counts = Counter(flat)
        most = [v for v, _ in counts.most_common(max_per_dim)]
        dim_values[d] = most

    # Build node index
    labels = []
    label_to_index = {}
    dim_node_range = {}
    for d in dims:
        start = len(labels)
        for v in dim_values[d]:
            lbl = f"{v}"
            label_to_index[(d, v)] = len(labels)
            labels.append(lbl)
        end = len(labels) - 1 if len(labels) > start else start
        dim_node_range[d] = (start, end)

    # Build links between adjacent selected dimensions
    source_idx = []
    target_idx = []
    values = []

    for i in range(len(dims) - 1):
        d1 = dims[i]
        d2 = dims[i + 1]

        pair_counts = Counter()
        for _, row in df.iterrows():
            a = normalize_cell(row.get(d1, ""))
            b = normalize_cell(row.get(d2, ""))
            # expand comma lists
            a_items = [x.strip() for x in a.split(",")] if a else []
            b_items = [x.strip() for x in b.split(",")] if b else []
            for ai in a_items:
                for bi in b_items:
                    if ai and bi and ai in dim_values[d1] and bi in dim_values[d2]:
                        pair_counts[(ai, bi)] += 1

        for (ai, bi), cnt in pair_counts.items():
            if cnt >= min_link_value:
                s = label_to_index[(d1, ai)]
                t = label_to_index[(d2, bi)]
                source_idx.append(s)
                target_idx.append(t)
                values.append(cnt)

    # Node colors by dimension
    import colorsys

    colors = []
    base_colors = {}
    for idx, d in enumerate(dims):
        h = (idx / max(1, len(dims)))
        r, g, b = colorsys.hsv_to_rgb(h, 0.5, 0.9)
        hexc = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        base_colors[d] = hexc

    for d in dims:
        for _ in dim_values[d]:
            colors.append(base_colors[d])

    # Build sankey figure
    node = dict(label=labels, pad=20, thickness=18, color=colors)
    link = dict(source=source_idx, target=target_idx, value=values)

    fig = go.Figure(data=[go.Sankey(node=node, link=link, arrangement='snap')])
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=700)
    return fig


st.title("NodeSynth — Taxonomy Viewer")

with st.sidebar:
    st.header("Controls")
    st.markdown("Select dimensions to display and press **Generate Taxonomy**")

    csv_path = st.text_input("CSV path", value="NodeSynth_Data_med_Full_Export.csv")

    all_dims = [
        ("Domain", "Domain"),
        ("level1", "level1"),
        ("level2", "level2"),
        ("level3", "level3"),
        ("user_case", "user_case"),
        ("user_group", "user_group"),
        ("extracted_occupations", "extracted_occupations"),
        ("extracted_Demographics", "extracted_Demographics"),
        ("extracted_Country", "extracted_Country"),
    ]

    # checkboxes for each dim
    selected_dims = []
    st.write("**Dimensions**")
    for display, col in all_dims:
        if st.checkbox(display, value=col in ["Domain", "level1", "level2", "level3", "user_group", "extracted_Country"]):
            selected_dims.append(col)

    max_nodes = st.slider("Max nodes per dimension", min_value=5, max_value=200, value=80)
    min_link = st.slider("Min link count to show", min_value=1, max_value=20, value=1)

    generate = st.button("Generate Taxonomy", type="primary")

if generate:
    try:
        df = load_data(csv_path)
    except Exception as e:
        st.error(f"Failed loading CSV: {e}")
        st.stop()

    # Validate selected dims exist
    missing = [d for d in selected_dims if d not in df.columns]
    if missing:
        st.error(f"The following selected columns are not in CSV: {missing}")
        st.write("CSV columns:", list(df.columns))
        st.stop()

    if len(selected_dims) < 2:
        st.warning("Select at least two dimensions to build links.")
        st.stop()

    fig = build_sankey(df, selected_dims, max_per_dim=max_nodes, min_link_value=min_link)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show data sample and counts"):
        st.write(df[selected_dims].head(50))
        # show counts for top nodes per dim
        for d in selected_dims:
            vals = df[d].dropna().map(normalize_cell)
            flat = []
            for v in vals:
                if "," in v:
                    flat.extend([p.strip() for p in v.split(",") if p.strip()])
                elif v:
                    flat.append(v)
            c = Counter(flat)
            st.write(f"Top values for {d}:", c.most_common(10))

    st.success("Taxonomy generated — toggle dimensions in the sidebar and regenerate to update.")

