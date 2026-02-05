import streamlit as st

# Configure page
st.set_page_config(page_title="NodeSynth", page_icon="🔗", layout="wide")

# Header
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.markdown("### 🔗 NodeSynth")
with col2:
    st.markdown("<div style='text-align: right; padding-top: 10px;'>Synthetic Data & Eval Prototype</div>", unsafe_allow_html=True)

st.divider()

# Sidebar navigation
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("""
    - 🎯 **Concept** ← Active
    - 🌳 Taxonomy
    - 📊 Data
    - ✓ Evaluate
    - 📈 Analyze
    """)

# Initialize session state for active view
if "active_view" not in st.session_state:
    st.session_state.active_view = None

# Main layout: left sidebar (1/4) and right content (3/4)
left_col, right_col = st.columns([1, 3])

# Left sidebar with buttons
with left_col:
    st.markdown("### Views")
    if st.button("🎯 Concept", use_container_width=True, key="concept_btn"):
        st.session_state.active_view = "concept"

# Right content area
with right_col:
    if st.session_state.active_view == "concept":
        tab1 = st.tabs(["Concept"])[0]
        with tab1:
            st.markdown("## Configure Dataset")
            st.markdown("Define the scope, constraints, and target context for your synthetic data generation.")

            st.markdown("")

            # Form layout
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### Target Concept")
                target_concept = st.text_input(
                    "Target Concept",
                    value="Cultural Representation",
                    label_visibility="collapsed",
                    disabled=False
                )

            with col2:
                pass

            st.markdown("### Description & Context")
            description = st.text_area(
                "Description & Context",
                value="Depiction, portrayal, or symbolization of cultures, identities, and experiences",
                height=100,
                label_visibility="collapsed"
            )

            st.markdown("")

            # Two column layout for countries and languages
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### Target Countries")
                target_countries = st.text_input(
                    "Target Countries",
                    placeholder="Enter countries...",
                    label_visibility="collapsed"
                )
                st.markdown('<span style="color: #6366f1;">Global ✕</span>', unsafe_allow_html=True)

            with col2:
                st.markdown("### Target Languages")
                target_languages = st.text_input(
                    "Target Languages",
                    placeholder="Enter languages...",
                    label_visibility="collapsed"
                )
                st.markdown('<span style="color: #6366f1;">English ✕</span>', unsafe_allow_html=True)

            st.markdown("")

            # Use Case and Modality
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### Use Case")
                use_case = st.text_input(
                    "Use Case",
                    value="Advice seeking",
                    label_visibility="collapsed"
                )

            with col2:
                st.markdown("### Modality")
                modality = st.text_input(
                    "Modality",
                    placeholder="Select modality...",
                    label_visibility="collapsed"
                )

            st.markdown("")

            # Generate Taxonomy button
            st.button(
                "Generate Taxonomy",
                use_container_width=True,
                type="primary"
            )
    else:
        st.markdown("")
