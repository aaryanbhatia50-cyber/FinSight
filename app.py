import streamlit as st

from src.pipeline.hybrid_pipeline import analyze_customer_query
from src.utils.spelling import find_spelling_errors


st.set_page_config(
    page_title="FinSight",
    page_icon="💬",
    layout="centered"
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 28px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 28px;
        margin-bottom: 18px;
    }

    [data-testid="stTextInput"] input {
        height: 52px;
        padding: 0 14px;
        font-size: 16px;
        line-height: 52px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">💬 FinSight</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent Customer Support Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Get instant insights into your banking customer support queries."
)

st.divider()


with st.form("support_form"):

    customer_query = st.text_input(
        "Describe your problem",
        placeholder="Example: I can't verify my identity."
    )

    analyze_button = st.form_submit_button(
        "Analyze Query",
        use_container_width=True
    )


if analyze_button:

    if not customer_query.strip():

        st.warning(
            "Please describe your problem first."
        )

    else:

        spelling_errors = find_spelling_errors(
            customer_query
        )

        if spelling_errors:

            st.warning(
                "Possible spelling error detected. Please check your query."
            )

        else:

            try:

                with st.spinner("Analyzing your query..."):

                    result = analyze_customer_query(
                        customer_query
                    )

                if result["route"] == "OUT_OF_SCOPE":

                    st.info(
                        "This query is outside the scope of FinSight. "
                        "Please enter a banking-related customer support issue."
                    )

                elif result["route"] == "ML":

                    st.markdown(
                        '<div class="section-title">Issue Identified</div>',
                        unsafe_allow_html=True
                    )

                    intent = (
                        result["intent"]
                        .replace("_", " ")
                        .title()
                    )

                    st.write(
                        f"Your query appears to be related to **{intent}**."
                    )

                else:

                    analysis = result["llm_analysis"]

                    st.markdown(
                        '<div class="section-title">Support Response</div>',
                        unsafe_allow_html=True
                    )

                    st.info(
                        analysis.customer_response
                    )

            except Exception:

                st.error(
                    "We couldn't analyze your query right now. "
                    "Please try again."
                )