import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate budget allocation
def calculate_budget(income, adults, children):
    essential_expenses = 0.5 * income  # 50% of income
    savings = 0.2 * income  # 20% of income
    discretionary_spending = 0.3 * income  # 30% of income
    
    per_adult = essential_expenses * 0.6 / max(adults, 1)
    per_child = essential_expenses * 0.4 / max(children, 1) if children > 0 else 0
    
    return essential_expenses, savings, discretionary_spending, per_adult, per_child

# Streamlit UI
st.title("💰 Family Budget Tracker & Financial Planner")
st.write("Plan your family's finances efficiently based on income and household members.")

# User Inputs
income = st.number_input("Enter Monthly Income ($)", min_value=0, value=5000, step=100)
adults = st.number_input("Number of Adults", min_value=1, value=2, step=1)
children = st.number_input("Number of Children", min_value=0, value=2, step=1)

if st.button("Calculate Budget"):
    essential, savings, discretionary, per_adult, per_child = calculate_budget(income, adults, children)
    
    st.subheader("📊 Budget Allocation")
    st.write(f"**Essential Expenses:** ${essential:,.2f}")
    st.write(f"**Savings:** ${savings:,.2f}")
    st.write(f"**Discretionary Spending:** ${discretionary:,.2f}")
    
    st.subheader("👨‍👩‍👧‍👦 Per Person Allocation")
    st.write(f"**Per Adult Essential Expense:** ${per_adult:,.2f}")
    if children > 0:
        st.write(f"**Per Child Essential Expense:** ${per_child:,.2f}")
    
    # Pie Chart Visualization
    labels = ['Essential Expenses', 'Savings', 'Discretionary Spending']
    values = [essential, savings, discretionary]
    
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
    ax.set_title("Family Budget Allocation")
    st.pyplot(fig)
    
    st.success("✔ Budget Plan Generated Successfully!")
