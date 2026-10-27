
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# 1. Page setup
st.set_page_config(
    page_title="ShopSense AI",
    page_icon="🛍️",
    layout="wide"
)

# 2. Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# 3. Simple dark-blue design
st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #E6EDF7;
}
[data-testid="stSidebar"] {
    background-color: #17243A;
}
[data-testid="stMetric"] {
    background-color: #1B2A41;
    border: 1px solid #304764;
    padding: 15px;
    border-radius: 12px;
}
h1, h2, h3 {
    color: #F0F6FF !important;
}
.stButton > button {
    background-color: #2476D2;
    color: white;
    border-radius: 8px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# 4. Load existing CSV files
customers = pd.read_csv(DATA / "customers.csv")
orders = pd.read_csv(DATA / "orders.csv")
items = pd.read_csv(DATA / "order_items.csv")
products = pd.read_csv(DATA / "product_summary.csv")
rfm = pd.read_csv(DATA / "customer_rfm.csv")
segments = pd.read_csv(DATA / "customer_segments.csv")

orders["order_date"] = pd.to_datetime(
    orders["order_date"], errors="coerce"
)

# 5. Sidebar navigation
st.sidebar.title("🛍️ ShopSense AI")
st.sidebar.caption("Customer Intelligence Platform")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Customer Segmentation",
        "Purchase Prediction",
        "Product Recommendations"
    ]
)

st.sidebar.divider()
st.sidebar.caption("E-Commerce Analytics")

# 6. Main heading
st.title("🛍️ ShopSense AI")
st.caption(
    "Understand customers. Predict purchases. Discover products."
)

# ==================================================
# PAGE 1: OVERVIEW
# ==================================================
if page == "Overview":

    st.subheader("Business Overview")

    a, b, c, d = st.columns(4)

    a.metric("Customers", f"{customers['customer_id'].nunique():,}")
    b.metric("Total Orders", f"{orders['order_id'].nunique():,}")
    c.metric("Products", f"{products['product_id'].nunique():,}")
    d.metric("Customer Segments", f"{segments['Segment'].nunique():,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Top Products")

        top_products = products.sort_values(
            "units_sold", ascending=False
        ).head(8)

        st.bar_chart(
            top_products.set_index("product_name")["units_sold"],
            horizontal=True
        )

    with right:
        st.subheader("Order Status")

        st.bar_chart(orders["order_status"].value_counts())

    st.subheader("Monthly Orders")

    monthly_orders = (
        orders.dropna(subset=["order_date"])
        .set_index("order_date")
        .resample("ME")["order_id"]
        .nunique()
    )

    st.line_chart(monthly_orders)

# ==================================================
# PAGE 2: CUSTOMER SEGMENTATION
# ==================================================
elif page == "Customer Segmentation":

    st.subheader("Customer Segmentation")
    st.write("Explore customer groups created using RFM and K-Means.")

    st.markdown("### Customer Count by Segment")

    segment_counts = segments["Segment"].value_counts().sort_index()
    st.bar_chart(segment_counts)

    st.markdown("### Customer Details")
    st.dataframe(segments.head(100), width="stretch", hide_index=True)

    st.download_button(
        "Download Segment Data",
        segments.to_csv(index=False).encode("utf-8"),
        file_name="customer_segments.csv",
        mime="text/csv"
    )

    with st.expander("What is RFM?"):
        st.write("Recency: How recently a customer purchased.")
        st.write("Frequency: How often a customer purchased.")
        st.write("Monetary: How much a customer spent.")

# ==================================================
# PAGE 3: PURCHASE PREDICTION
# ==================================================
elif page == "Purchase Prediction":

    st.subheader("Repeat-Purchase Prediction")
    st.write("Select a customer to estimate repeat-purchase likelihood.")

    customer_ids = rfm["customer_id"].dropna().astype(str).unique()

    customer_id = st.selectbox(
        "Select Customer",
        customer_ids
    )

    if st.button("Predict Repeat Purchase", width="stretch"):

        model_path = MODELS / "repeat_purchase_model_compressed.pkl"
        features_path = MODELS / "model_features.pkl"

        if not model_path.exists() or not features_path.exists():
            st.error("The trained model or its feature list is missing.")

        else:
            # Load the saved model and its original feature names
            model = joblib.load(model_path)
            feature_names = joblib.load(features_path)

            customer_data = rfm[
                rfm["customer_id"].astype(str) == customer_id
            ]

            # Get this customer's delivered order history
            delivered = orders[
                orders["order_status"] == "Delivered"
            ]

            customer_orders = delivered[
                delivered["customer_id"].astype(str) == customer_id
            ]

            if customer_data.empty:
                st.error("Customer data was not found.")

            else:
                # Start with the customer's RFM values
                values = customer_data.iloc[0].to_dict()

                values["order_count"] = orders[
    (orders["customer_id"].astype(str) == customer_id) &
    (orders["order_status"] == "Delivered")
]["order_id"].nunique()

                if not customer_orders.empty:
                    first_order = customer_orders["order_date"].min()
                    last_order = customer_orders["order_date"].max()
                    latest_date = delivered["order_date"].max()

                    values["recency_days"] = (
                        latest_date - last_order
                    ).days

                    values["customer_age_days"] = (
                        latest_date - first_order
                    ).days

                    values["delivered_orders"] = (
                        customer_orders["order_id"].nunique()
                    )

                # Use the same feature names as model training
                X = pd.DataFrame([{
                    feature: values[feature]
                    for feature in feature_names
                    if feature in values
                }])

                missing = [
                    feature for feature in feature_names
                    if feature not in X.columns
                ]

                if missing:
                    st.error(
                        "Training features are missing: "
                        + ", ".join(missing)
                        + ". Match these with notebook 05."
                    )

                else:
                    X = X[feature_names]

                    prediction = model.predict(X)[0]

                    if prediction == 1:
                        st.success("Repeat purchase is predicted.")
                    else:
                        st.info("Repeat purchase is not predicted.")

                    if hasattr(model, "predict_proba"):
                        probability = model.predict_proba(X)[0]
                        classes = list(model.classes_)

                        if 1 in classes:
                            score = probability[classes.index(1)]
                            st.metric(
                                "Estimated Repeat-Purchase Probability",
                                f"{score:.1%}"
                            )
                            st.progress(float(score))

                    st.caption(
                        "Prediction is an estimate, not a guarantee."
                    )

# ==================================================
# PAGE 4: PRODUCT RECOMMENDATIONS
# ==================================================
elif page == "Product Recommendations":

    st.subheader("Product Recommendations")
    st.write("Find popular products that a customer has not purchased.")

    customer_ids = orders["customer_id"].dropna().astype(str).unique()

    customer_id = st.selectbox(
        "Select Customer",
        customer_ids,
        key="recommend_customer"
    )

    number_of_products = st.slider(
        "Number of Recommendations", 5, 20, 10
    )

    if st.button("Generate Recommendations", width="stretch"):

        # Keep delivered orders for this customer
        delivered = orders[
            orders["order_status"] == "Delivered"
        ]

        customer_orders = delivered[
            delivered["customer_id"].astype(str) == customer_id
        ]

        # Find the products already purchased
        customer_items = items[
            items["order_id"].isin(customer_orders["order_id"])
        ]

        bought_products = customer_items["product_id"].unique()

        # Remove those products from the product list
        recommendations = products[
            ~products["product_id"].isin(bought_products)
        ]

        # Recommend products with the highest sales
        recommendations = recommendations.sort_values(
            "units_sold", ascending=False
        ).head(number_of_products)

        if recommendations.empty:
            st.info("No recommendations are available for this customer.")

        else:
            st.success(
                f"{len(recommendations)} recommendations generated."
            )

            st.dataframe(
                recommendations,
                hide_index=True,
                width="stretch"
            )

            st.download_button(
                "Download Recommendations",
                recommendations.to_csv(index=False).encode("utf-8"),
                file_name="product_recommendations.csv",
                mime="text/csv"
            )

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("ShopSense AI | E-Commerce Customer Intelligence")
