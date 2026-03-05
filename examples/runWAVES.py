from copy import deepcopy
from time import perf_counter
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from waves import Project
from waves.utilities import load_yaml

# Update core Pandas display settings
#pd.options.display.float_format = "{:,.2f}".format
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

metrics_configuration = {
    "# Turbines": {"metric": "n_turbines"},
    "Turbine Rating (MW)": {"metric": "turbine_rating"},
    "Project Capacity (MW)": {
        "metric": "capacity",
        "kwargs": {"units": "mw"}
    },
    "# OSS": {"metric": "n_substations"},
    "Total Export Cable Length (km)": {"metric": "export_system_total_cable_length"},
    "Total Array Cable Length (km)": {"metric": "array_system_total_cable_length"},
    "CapEx ($)": {"metric": "capex"},
    "CapEx per kW ($/kW)": {
        "metric": "capex",
        "kwargs": {"per_capacity": "kw"}
    },
    "OpEx ($)": {"metric": "opex"},
    "OpEx per kW ($/kW)": {"metric": "opex", "kwargs": {"per_capacity": "kw"}},
    "AEP (MWh)": {
        "metric": "energy_production",
        "kwargs": {"units": "mw", "aep": True}
    },
    "AEP per kW (MWh/kW)": {
        "metric": "energy_production",
        "kwargs": {"units": "mw", "per_capacity": "kw", "aep": True}
    },
    "Net Capacity Factor With All Losses (%)": {
        "metric": "capacity_factor",
        "kwargs": {"which": "net"}
    },
    "Gross Capacity Factor (%)": {
        "metric": "capacity_factor",
        "kwargs": {"which": "gross"}
    },
    "Energy Availability (%)": {
        "metric": "availability",
        "kwargs": {"which": "energy"}
    },
    "LCOE ($/MWh)": {"metric": "lcoe"},
}


# Define the final order of the metrics in the resulting dataframes
metrics_order = [
    "# Turbines",
    "Turbine Rating (MW)",
    "Project Capacity (MW)",
    "# OSS",
    "Total Export Cable Length (km)",
    "Total Array Cable Length (km)",
    "FCR (%)",
    "Offtake Price ($/MWh)",
    "CapEx ($)",
    "CapEx per kW ($/kW)",
    "OpEx ($)",
    "OpEx per kW ($/kW)",
    "Annual OpEx per kW ($/kW)",
    "Energy Availability (%)",
    "Gross Capacity Factor (%)",
    "Net Capacity Factor With All Losses (%)",
    "AEP (MWh)",
    "AEP per kW (MWh/kW)",
    "LCOE ($/MWh)",
    "Potential AEP from WOMBAT (kWh)",
    "Production AEP from WOMBAT (kWh)",
]

capex_order = [
    "Array System",
    "Export System",
    "Offshore Substation",
    "Substructure",
    "Scour Protection",
    "Mooring System",
    "Turbine",
    "Array System Installation",
    "Export System Installation",
    "Offshore Substation Installation",
    "Substructure Installation",
    "Scour Protection Installation",
    "Mooring System Installation",
    "Turbine Installation",
    "Soft",
    "Project",
]

def run_waves(project_floating):
    start2 = perf_counter()
    project_floating.run(full_wind_rose=False)
    project_floating.wombat.env.cleanup_log_files()  # Delete logging data from the WOMBAT simulations
    end2 = perf_counter()
    
    print("-" * 29)  # separate our timing from the ORBIT and FLORIS run-time warnings
    print(f"Floating run time: {end2 - start2:,.2f} seconds")

    return project_floating

def average_and_save(dfs, filename, index_cols=None):
    df_concat = pd.concat(dfs)
    if index_cols:
        df_avg = df_concat.groupby(index_cols).mean()
    else:
        df_avg = df_concat.groupby(level=0).mean()
    df_avg.to_csv(filename)
    print(f"Saved: {filename}")

def runRunWAVES(NUM_RUNS, library_path):

    rng = np.random.default_rng(seed=834)

    capex_dfs = []
    opex_dfs = []
    charter_days_dfs = []
    mobilization_dfs = []
    delay_dfs = []
    failure_cost_dfs = []
    equipment_cost_dfs = []
    report_dfs = []
    losses_dfs = []
    df_projects = []

    config_floating = load_yaml(library_path / "project/config", "base_floating_deep.yaml")
    config_floating["floris_config"] = load_yaml(library_path / "project/config", config_floating["floris_config"])
    config_floating["floris_config"]["farm"]["turbine_library_path"] = library_path / "turbines"
    config_floating.update({"library_path": library_path})

    config_wombat = load_yaml(library_path / "project/config", config_floating["wombat_config"])

    for i in range(NUM_RUNS):
        print(f"\nRun {i + 1} of {NUM_RUNS}")

        # Load the project
        config = deepcopy(config_floating)
        config_wombat["random_generator"] = rng
        config_floating["wombat_config"] = config_wombat
        
        start = perf_counter()
        project_floating = Project.from_dict(config_floating)
        end = perf_counter()
        print(f"Floating loading time: {end - start:,.2f} seconds")

        # Run simulation
        project_floating = run_waves(project_floating)

        # Load key objects
        ev = project_floating.wombat.metrics.events
        years = project_floating.wombat.env.simulation_years
        metrics = project_floating.wombat.metrics
        materials = metrics.component_costs("project", by_category=True, by_task=True, by_action=False)
        avg_materials = materials[["materials_cost"]] / years

        # 0. CapEx Breakdown
        df_capex_floating = pd.DataFrame(
            project_floating.orbit.capex_detailed_soft_capex_breakdown.items(),
            columns=["Component", "CapEx ($) - Floating"]
        )
        df_capex_floating["CapEx ($/kW) - Floating"] = (
            df_capex_floating["CapEx ($) - Floating"] / project_floating.capacity("kw")
        )

        # Extract the onshore substation cost
        onshore_substation_cost = (
            project_floating.orbit.phases["ElectricalDesign"]
            .detailed_output["export_system"]["onshore_substation_costs"]
        )
        onshore_substation_cost_per_kw = onshore_substation_cost / project_floating.capacity("kw")

        # Append onshore substation as a row
        df_capex_floating.loc[len(df_capex_floating)] = [
            "Onshore Substation",
            onshore_substation_cost,
            onshore_substation_cost_per_kw
        ]

        # Set index for consistent merging
        df_capex_floating.set_index("Component", inplace=True)
        capex_dfs.append(df_capex_floating)
        
        # 1. Annual OpEx
        opex_df = metrics.opex(frequency='annual', by_category=True)
        opex_dfs.append(opex_df)

        # 2. Average Charter Days
        average_charter_days = []
        for name, vessel in project_floating.wombat.service_equipment.items():
            if vessel.settings.onsite or "TOW" in [el.value for el in vessel.settings.capability]:
                continue
            mobilizations = ev.loc[
                (ev.action.eq("mobilization") & ev.reason.str.contains("arrived on site"))
                & ev.agent.eq(name),
                ["agent", "env_time"]
            ]
            leaving = ev.loc[
                ev.action.eq("leaving site")
                & ev.agent.eq(name),
                ["agent", "env_time"]
            ]
            if mobilizations.shape[0] - leaving.shape[0] == 1:
                mobilizations = mobilizations.iloc[:-1]
            charter_days = (leaving.env_time.values - mobilizations.env_time.values) / 24
            average_charter_days.append([name, charter_days.mean()])
        charter_days_df = pd.DataFrame(average_charter_days, columns=["vessel", "average charter days"]).set_index("vessel")
        charter_days_dfs.append(charter_days_df)

        # 3. Mobilization Summary
        mobilization_summary = (
            ev.loc[ev.action.eq("mobilization") & ev.duration.gt(0), ["agent", "duration"]]
            .groupby("agent")
            .count()
            .rename(columns={"duration": "mobilizations"})
            .join(
                ev.loc[ev.action.eq("mobilization"), ["agent", "duration", "equipment_cost"]]
                .groupby("agent")
                .sum()
            )
        )
        mobilization_summary.duration /= 24
        mobilization_dfs.append(mobilization_summary)

        # 4. Delay Summary
        delay_summary = (
            ev.loc[
                ev.agent.isin(project_floating.wombat.service_equipment)
                & ev.duration.gt(0)
                & ev.action.eq("delay"),
                ["agent", "additional", "duration"]
            ]
            .replace({
                "no work requests submitted by start of shift": "no requests",
                "no work requests, waiting until the next shift": "no requests",
                "weather unsuitable to transfer crew": "weather delay",
                "work shift has ended; waiting for next shift to start": "end of shift",
                "insufficient time to complete travel before end of the shift": "end of shift",
                "will return next year": "end of charter",
            })
            .groupby(["agent", "additional"])
            .sum()
            .reset_index(drop=False)
            .set_index(["agent", "additional"])
            / 24
        )
        delay_dfs.append(delay_summary)

        # 5. Failure Costs
        timing = metrics.process_times()[["N"]].rename(columns={"N": "annual_occurrences"}) / years
        average_failures_costs = (
            avg_materials
            .rename(columns={"materials_cost": "annual_materials_cost"})
            .join(timing, how="outer")
            .fillna(0.0)
        )
        failure_cost_dfs.append(average_failures_costs)

        # 6. Equipment Cost Summary
        equipment_cost_df = metrics.equipment_costs(frequency="annual", by_equipment=True)
        equipment_cost_dfs.append(equipment_cost_df)

        # 7. Report DF
        project_name_floating = "Standardized Moorings Case - Floating"
        report_df_floating = project_floating.generate_report(metrics_configuration, project_name_floating).T
        n_years_floating = project_floating.operations_years
        additional_reporting = pd.DataFrame(
            [
                ["FCR (%)", project_floating.fixed_charge_rate],
                ["Offtake Price ($/MWh)", project_floating.offtake_price],
                [
                    "Annual OpEx per kW ($/kW)",
                    report_df_floating.loc["OpEx per kW ($/kW)", project_name_floating] / n_years_floating
                ],
                ["Potential AEP from WOMBAT (kWh)", project_floating.wombat.metrics.potential.windfarm.values.sum()/n_years_floating],
                ["Production AEP from WOMBAT (kWh)", project_floating.wombat.metrics.production.windfarm.values.sum()/n_years_floating],
            ],
            columns=["Project"] + report_df_floating.columns.tolist(),
        ).set_index("Project")

        report_df_floating = pd.concat((report_df_floating, additional_reporting), axis=0)
        report_df_floating.index.name = "Metrics"
        report_df_floating.loc[report_df_floating.index.str.contains("%")] *= 100
        report_dfs.append(report_df_floating)

        df_projects.append(generate_report_lcoe_breakdown_adjusted(project_floating))

        #8. Losses report
        report_df_losses = project_floating.loss_ratio(breakdown=True)
        losses_dfs.append(report_df_losses) 
    
    df_capex = pd.concat(capex_dfs).groupby("Component").mean()
    df_opex = pd.concat(opex_dfs).groupby(level=0).mean()
    df_report = pd.concat(report_dfs).groupby("Metrics").mean()
    
    capex_per_kw = df_capex[["CapEx ($/kW) - Floating"]]
    capex_metrics = capex_per_kw.copy()
    capex_metrics.index = [f"CapEx - {component} ($/kW)" for component in capex_per_kw.index]
    project_col_name = df_report.columns[0]
    capex_metrics.columns = [project_col_name]
    df_report = pd.concat([df_report, capex_metrics])
    df_report.to_csv(library_path / 'results/lcoe_results.csv')

    return df_projects, df_report






def generate_report_lcoe_breakdown(project) -> pd.DataFrame:
    """Generates a dataframe containing the detailed breakdown of LCOE (Levelized Cost of
    Energy) metrics for the project, which is used to produce LCOE waterfall charts and
    CapEx donut charts in the Cost of Wind Energy Review. The breakdown includes the
    contributions of each CapEx and OpEx component (from ORBIT and WOMBAT) to the LCOE in
    $/MWh.

    This function calculates the LCOE by considering both CapEx (from ORBIT) and OpEx (from
    WOMBAT),and incorporates the fixed charge rate (FCR) and net annual energy production
    (net AEP) into the computation for each component.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the detailed LCOE breakdown with the following columns:
            - "Component": The name of the project component (e.g., "Turbine", "Balance
                of System CapEx", "OpEx").
            - "Category": The category of the component (e.g., "Turbine", "Balance of System
                CapEx", "Financial CapEx", "OpEx").
            - "Value ($/kW)": The value of the component in $/kW.
            - "Fixed charge rate (FCR) (real)": The real fixed charge rate (FCR) applied to the
                component.
            - "Value ($/kW-yr)": The value of the component in $/kW-yr, after applying the FCR.
            - "Net AEP (MWh/kW/yr)": The net annual energy production (AEP) in MWh/kW/yr.
            - "Value ($/MWh)": The value of the component in $/MWh, calculated by dividing the
                $/kW-yr value by the net AEP.

    Notes
    -----
    - CapEx components are categorized into "Turbine", "Balance of System CapEx", and "Financial
        CapEx".
    - OpEx components are derived from WOMBAT's OpEx metrics, categorized as "OpEx".
    - The LCOE is calculated by considering both CapEx and OpEx components, and adjusting for
        net AEP and FCR.
    - Rows with a value of 0 in the "Value ($/MWh)" column are removed to avoid clutter in
        annual reporting charts.
    """
    # Static values
    fcr = project.fixed_charge_rate
    net_aep = project.energy_production(units="mw", per_capacity="kw", aep=True)

    # Handle CapEx outputs from ORBIT
    try:
        capex_data = project.orbit.capex_detailed_soft_capex_breakdown_per_kw
    except AttributeError:
        capex_data = project.orbit.capex_breakdown_per_kw

    turbine_components = ("Turbine", "Nacelle", "Blades", "Tower", "RNA")
    financial_components = (
        "Construction",
        "Decommissioning",
        "Financing",
        "Contingency",
        "Soft",
    )
    columns = [
        "Component",
        "Category",
        "Value ($/kW)",
        "Fixed charge rate (FCR) (real)",
        "Value ($/kW-yr)",
        "Net AEP (MWh/kW/yr)",
        "Value ($/MWh)",
    ]

    df = pd.DataFrame.from_dict(capex_data, orient="index", columns=["Value ($/kW)"])
    df["Category"] = "BOS"
    df.loc[df.index.isin(turbine_components), "Category"] = "Turbine"
    df.loc[df.index.isin(financial_components), "Category"] = "Financial CapEx"
    df.Category = df.Category.str.replace("BOS", "Balance of System CapEx")

    df["Fixed charge rate (FCR) (real)"] = fcr
    df["Value ($/kW-yr)"] = df["Value ($/kW)"] * fcr
    df["Value ($/MWh)"] = df["Value ($/kW-yr)"] / net_aep
    df["Net AEP (MWh/kW/yr)"] = net_aep
    df = df.reset_index(drop=False)
    df = df.rename(columns={"index": "Component"})

    # Handle OpEx outputs from WOMBAT
    opex = (
        project.wombat.metrics.opex(frequency="annual", by_category=True)
        .mean(axis=0)
        .to_frame("Value ($/kW-yr)")
        .join(
            project.wombat.metrics.opex(frequency="annual", by_category=True)
            .sum(axis=0)
            .to_frame("Value ($/kW)")
        )
        .drop("OpEx")
    )
    opex /= project.capacity("kw")
    opex.index = opex.index.str.replace("_", " ").str.title()
    opex.index.name = "Component"
    opex["Category"] = "OpEx"
    opex["Fixed charge rate (FCR) (real)"] = fcr
    opex["Net AEP (MWh/kW/yr)"] = net_aep
    opex["Value ($/MWh)"] = opex["Value ($/kW-yr)"] / net_aep
    opex = opex.reset_index(drop=False)[columns]

    # Concatenate CapEx and OpEx rows
    df = pd.concat((df, opex)).reset_index(drop=True).reset_index(names=["Original Order"])

    # Define the desired order of categories for sorting
    order_of_categories = ["Turbine", "Balance of System CapEx", "Financial CapEx", "OpEx"]

    # Sort the dataframe based on the custom category order
    df["Category"] = pd.Categorical(
        df["Category"], categories=order_of_categories, ordered=True
    )
    df = (
        df.sort_values(by=["Category", "Original Order"])
        .drop(columns=["Original Order"])
        .reset_index(drop=True)
    )

    # Remove rows where 'Value ($/MWh)' is zero to avoid 0 values on annual reporting charts
    df = df[df["Value ($/MWh)"] != 0]

    # Re-order the columns so that it is more intuitive for the analyst
    df = df[
        [
            "Component",
            "Category",
            "Value ($/kW)",
            "Fixed charge rate (FCR) (real)",
            "Value ($/kW-yr)",
            "Net AEP (MWh/kW/yr)",
            "Value ($/MWh)",
        ]
    ]

    # Reset index and return the dataframe
    df = df.reset_index(drop=True)
    return df




def generate_report_lcoe_breakdown_adjusted(project):
    """Generates a dataframe containing the detailed breakdown of LCOE (Levelized Cost of
    Energy) metrics for the project, which is used to produce LCOE waterfall charts and
    CapEx donut charts in the Cost of Wind Energy Review. The breakdown includes the
    contributions of each CapEx and OpEx component (from ORBIT and WOMBAT) to the LCOE in
    $/MWh.

    This function calculates the LCOE by considering both CapEx (from ORBIT) and OpEx (from
    WOMBAT),and incorporates the fixed charge rate (FCR) and net annual energy production
    (net AEP) into the computation for each component.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the detailed LCOE breakdown with the following columns:
            - "Component": The name of the project component (e.g., "Turbine", "Balance
                of System CapEx", "OpEx").
            - "Category": The category of the component (e.g., "Turbine", "Balance of System
                CapEx", "Financial CapEx", "OpEx").
            - "Value ($/kW)": The value of the component in $/kW.
            - "Fixed charge rate (FCR) (real)": The real fixed charge rate (FCR) applied to the
                component.
            - "Value ($/kW-yr)": The value of the component in $/kW-yr, after applying the FCR.
            - "Net AEP (MWh/kW/yr)": The net annual energy production (AEP) in MWh/kW/yr.
            - "Value ($/MWh)": The value of the component in $/MWh, calculated by dividing the
                $/kW-yr value by the net AEP.

    Notes
    -----
    - CapEx components are categorized into "Turbine", "Balance of System CapEx", and "Financial
        CapEx".
    - OpEx components are derived from WOMBAT's OpEx metrics, categorized as "OpEx".
    - The LCOE is calculated by considering both CapEx and OpEx components, and adjusting for
        net AEP and FCR.
    - Rows with a value of 0 in the "Value ($/MWh)" column are removed to avoid clutter in
        annual reporting charts.
    """
    # Static values
    fcr = project.fixed_charge_rate
    net_aep = project.energy_production(units="mw", per_capacity="kw", aep=True)

    # Handle CapEx outputs from ORBIT
    try:
        capex_data = project.orbit.capex_detailed_soft_capex_breakdown_per_kw
    except AttributeError:
        capex_data = project.orbit.capex_breakdown_per_kw

    turbine_components = ("Turbine", "Nacelle", "Blades", "Tower", "RNA")
    financial_components = (
        "Construction",
        "Decommissioning",
        "Financing",
        "Contingency",
        "Soft",
    )
    columns = [
        "Component",
        "Category",
        "Value ($/kW)",
        "Fixed charge rate (FCR) (real)",
        "Value ($/kW-yr)",
        "Net AEP (MWh/kW/yr)",
        "Value ($/MWh)",
    ]
    soft_cost_components = ("Construction Insurance", "Commissioning", "Procurement Contingency", "Installation Contingency",
                            "Construction Financing", "Decommissioning")
    '''
    df = pd.DataFrame.from_dict(capex_data, orient="index", columns=["Value ($/kW)"])
    df["Category"] = "BOS"
    df.loc[df.index.isin(turbine_components), "Category"] = "Turbine"
    df.loc[df.index.isin(financial_components), "Category"] = "Financial CapEx"
    df.Category = df.Category.str.replace("BOS", "Balance of System CapEx")
    '''
    df = pd.DataFrame.from_dict(capex_data, orient="index", columns=["Value ($/kW)"])
    df["Category"] = "CapEx"
    #df.loc[df.index.isin(turbine_components), "Category"] = "Turbine"
    #df.loc[df.index.isin(financial_components), "Category"] = "Financial CapEx"
    #df.Category = df.Category.str.replace("BOS", "CapEx")
    
    df = df.reset_index(drop=False)
    df = df.rename(columns={"index": "Component"})

    original_order = df["Component"].tolist()
    original_order.append("Soft Costs")
    original_order.insert(0, original_order.pop(original_order.index("Turbine")))
    
    df.loc[df["Component"].isin(soft_cost_components), "Category"] = "Soft Costs"
    df.loc[df["Component"].isin(soft_cost_components), "Component"] = "Soft Costs"
    df = df.groupby(["Component", "Category"], as_index=False).sum(numeric_only=True)
    df.loc[df["Component"] == "Soft Costs", "Category"] = "CapEx"

    df["Component"] = pd.Categorical(df["Component"], categories=original_order, ordered=True)
    df = df.sort_values("Component").reset_index(drop=True)

    df["Fixed charge rate (FCR) (real)"] = fcr
    df["Value ($/kW-yr)"] = df["Value ($/kW)"] * fcr
    df["Value ($/MWh)"] = df["Value ($/kW-yr)"] / net_aep
    df["Net AEP (MWh/kW/yr)"] = net_aep
    df = df.reset_index(drop=False)
    

    # Handle OpEx outputs from WOMBAT
    opex = (
        project.wombat.metrics.opex(frequency="annual", by_category=True)
        .mean(axis=0)
        .to_frame("Value ($/kW-yr)")
        .join(
            project.wombat.metrics.opex(frequency="annual", by_category=True)
            .sum(axis=0)
            .to_frame("Value ($/kW)")
        )
        .drop("OpEx")
    )
    opex /= project.capacity("kw")
    opex.index = opex.index.str.replace("_", " ").str.title()
    opex.index.name = "Component"
    opex["Category"] = "OpEx"
    opex["Fixed charge rate (FCR) (real)"] = fcr
    opex["Net AEP (MWh/kW/yr)"] = net_aep
    opex["Value ($/MWh)"] = opex["Value ($/kW-yr)"] / net_aep
    opex = opex.reset_index(drop=False)[columns]

    # Concatenate CapEx and OpEx rows
    df = pd.concat((df, opex)).reset_index(drop=True).reset_index(names=["Original Order"])

    # Define the desired order of categories for sorting
    order_of_categories = ["CapEx", "OpEx"]
    #order_of_categories = ["Turbine", "Balance of System CapEx", "Financial CapEx", "OpEx"]

    # Sort the dataframe based on the custom category order
    df["Category"] = pd.Categorical(
        df["Category"], categories=order_of_categories, ordered=True
    )
    df = (
        df.sort_values(by=["Category", "Original Order"])
        .drop(columns=["Original Order"])
        .reset_index(drop=True)
    )

    # Remove rows where 'Value ($/MWh)' is zero to avoid 0 values on annual reporting charts
    df = df[df["Value ($/MWh)"] != 0]

    # Re-order the columns so that it is more intuitive for the analyst
    df = df[
        [
            "Component",
            "Category",
            "Value ($/kW)",
            "Fixed charge rate (FCR) (real)",
            "Value ($/kW-yr)",
            "Net AEP (MWh/kW/yr)",
            "Value ($/MWh)",
        ]
    ]

    # Reset index and return the dataframe
    df = df.reset_index(drop=True)
    return df



def plot_LCOE_waterfall(technology, df, width=8, height=6, y_min=None, y_max=None, outfile=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df_final = df

    # Extract the final components, values, and categories
    components = df_final['Component']
    values = df_final['Value ($/MWh)']
    categories = df_final['Category']

    # Calculate total LCOE
    total_lcoe = values.sum()

    # Append 'Total LCOE' to components and values
    components = pd.concat([components, pd.Series('LCOE')], ignore_index=True)
    values = pd.concat([values, pd.Series(total_lcoe)], ignore_index=True)
    categories = pd.concat([categories, pd.Series('Total')], ignore_index=True)

    # Setup the figure and axes
    fig, ax = plt.subplots(figsize=(width, height))

    # Initial bar is set at 0
    bar_positions = np.arange(len(values))
    bar_values = values.tolist()
    bar_labels = components.tolist()

    # Waterfall plot values
    cumulative_values = np.cumsum([0] + bar_values[:-1])
    base = cumulative_values

    # Define color map for categories
    category_colors = {
        'CapEx': 'g',
        'OpEx': 'yellow',
        'Total': 'darkblue'
    }

    # Plotting the bars
    for i in range(len(bar_values)):
        color = category_colors.get(categories[i], 'grey')
        if i == len(bar_values) - 1:
            # Total LCOE bar is fully visible and dark blue
            ax.bar(bar_positions[i], bar_values[i], bottom=0, color=category_colors["Total"], edgecolor='black', label='LCOE', zorder=3)
        else:
            # Invisible base bar for intermediate bars
            ax.bar(bar_positions[i], base[i], bottom=0, color='white', edgecolor='white', zorder=1)
            # Intermediate bars
            ax.bar(bar_positions[i], bar_values[i], bottom=base[i], color=color, edgecolor='black', zorder=3)

    # Labeling bars with values
    for i, (pos, val) in enumerate(zip(bar_positions, bar_values)):
        alignment = 'center' if val > 0 else 'top'
        if i == len(bar_values) - 1:
            # Round total LCOE to integer for display
            ax.text(pos, val + 0.5, f'{int(round(val))}', ha='center', va='bottom', color='black', zorder=4)
        else:
            ax.text(pos, base[i] + val + 0.5, f'{val:.1f}', ha='center', va='bottom', color='black', zorder=4)

    # Add labels for each category above the bars
    category_positions = {}
    for i, (pos, val) in enumerate(zip(bar_positions, bar_values)):
        category = categories[i]
        if category not in category_positions:
            category_positions[category] = []
        category_positions[category].append((pos, val))

    # Determine a consistent height for the category labels
    label_y_position = base[2] * 0.6  # Default height for categories
    first_category_label_y_position = None

    for category, positions in category_positions.items():
        if category == 'Total':
            continue
        # Check if the technology string contains "DW"
        if "DW" in technology and len(positions) == 1:
            continue
        # Calculate the range for the category
        min_pos = min(pos for pos, _ in positions)
        max_pos = max(pos for pos, _ in positions)
        total_percentage = sum(val for _, val in positions)
        pct_of_total = (total_percentage / total_lcoe) * 100
        text_label = f'{category}\n({pct_of_total:.1f}%)'
        
        # Get the color of the current category and set text box background color
        color = category_colors.get(category, 'grey')
        bbox_props = dict(boxstyle="square,pad=0.3", edgecolor="black", facecolor=color)
        '''
        # Determine the height for the label
        if first_category_label_y_position is None:
            first_category_label_y_position = base[max_pos] + 1.75 * bar_values[max_pos]  # 20% above the height of the last bar in the first category
            y_position = first_category_label_y_position
        else:
        '''
        y_position = label_y_position
        
        # Place the label above the bars for this category at the determined height
        if color == "yellow":
            ax.text((min_pos + max_pos) / 2, y_position, text_label, ha='center', va='center', color='black', bbox=bbox_props, zorder=4)
        else:
            ax.text((min_pos + max_pos) / 2, y_position, text_label, ha='center', va='center', color='white', bbox=bbox_props, zorder=4)
    
    # Add horizontal grid lines behind the bars
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, zorder=0)

    # Add a grey horizontal line at the height of the total value
    #ax.axhline(total_lcoe, color='grey', linestyle='--', linewidth=1, zorder=1)
    
    # Add vertical grey lines to separate the bars from different categories
    if "DW" not in technology:
        category_boundaries = []
        last_category = categories[0]

        for i in range(1, len(categories)):
            if categories[i] != last_category:
                category_boundaries.append(i - 0.5)
                last_category = categories[i]

        for boundary in category_boundaries:
            ax.axvline(boundary, color='grey', linewidth=1.2, zorder=1)
    
    # Setting labels and title
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=45, ha='right')
    ax.set_ylabel('Levelized Cost of Energy (real, 2024$/MWh)')
    
    # Equal aspect ratio ensures that pie is drawn as a circle.
    if "DW" in technology:
        if "20" in technology:
            plt.title("Single-Turbine\nResidential (20 kW)", fontsize=16)
        if "100" in technology:
            plt.title("Single-Turbine\nCommercial (100 kW)", fontsize=16)
        if "1500" in technology:
            plt.title("Single-Turbine\nLarge (1,500 kW)", fontsize=16)

    # Set y-axis limits if specified
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    if y_max is not None:
        ax.set_ylim(top=y_max)

    # Tight layout for better spacing
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    #fig.savefig("C:\\Code\\WAVES\\library\\Standardized_Moorings\\results\\LCOE_waterfall.png", bbox_inches='tight')
    #fig.savefig("C:\\Code\\WAVES\\library\\Standardized_Moorings3\\results\\LCOE_waterfall.png", bbox_inches='tight')
    if outfile:
        fig.savefig(outfile, bbox_inches='tight')
    plt.show()



def create_waterfall_chart(df_1, total_1, df_2, total_2, color_2, outfile=None):
    import matplotlib.pyplot as plt
    import textwrap
    # Calculate the total value of df_1 and df_2
    total_df1 = df_1['Value ($/MWh)'].sum()
    total_df2 = df_2['Value ($/MWh)'].sum()
    
    # Get common CapEx components between df_1 and df_2
    common_components = set(df_1['Component']).intersection(df_2['Component'])
    common_categories = set(df_1['Category']).intersection(df_2['Category'])
    
    # Calculate differences in Value ($/kW) between df_1 and df_2 for common components
    '''
    differences = {comp: df_2[df_2['Component'] == comp]['Value ($/MWh)'].values[0] -
                            df_1[df_1['Component'] == comp]['Value ($/MWh)'].values[0]
                            for comp in common_components}
    '''
    # Sum values by category for each DataFrame
    df1_grouped = df_1.groupby("Category")["Value ($/MWh)"].sum()
    df2_grouped = df_2.groupby("Category")["Value ($/MWh)"].sum()
    # Find all unique categories present in either DataFrame
    all_categories = set(df1_grouped.index).union(df2_grouped.index)
    # Compute the difference for each category
    category_differences = {
        cat: df2_grouped.get(cat, 0) - df1_grouped.get(cat, 0)
        for cat in all_categories
    }
    
    # Sort the differences by absolute value
    sorted_diff = sorted(category_differences.items(), key=lambda x: abs(x[1]), reverse=True)
    sorted_diff = [(item[0], item[1]) for item in sorted_diff if item[1] != 0.0]
    
    # Initialize lists to store the heights and labels for the bars
    heights = [total_df1]
    labels = [total_1]
    white_bar_heights = [0]
    
    # Add the differences to the lists
    count = total_df1
    for comp, diff in sorted_diff:
        heights.append( - diff)
        labels.append(comp)
        white_bar_heights.append(count + diff)
        count = count + diff
    
    # Add the total of df_2 to the lists
    heights.append(total_df2)
    labels.append(total_2)
    white_bar_heights.append(0)
    
    # Plot the waterfall chart
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, white_bar_heights, color='white')

    colors = ['lightcoral' if h < 0 else 'lightgreen' for h in heights]
    colors[0] = color_2
    colors[-1] = color_2

    alphas = [1.0]*len(heights)
    alphas[0] = 0.6
    alphas[-1] = 0.6
    
    for i in range(len(heights)):
        ax.bar(labels[i], heights[i], bottom=white_bar_heights[i], color=colors[i], alpha=alphas[i])#, edgecolor='black', zorder = 50)
    ax.set_ylabel('LCOE (real, 2024$/MWh)')
    #plt.title('Waterfall Chart Comparison between df_1 and df_2')
    #ax.set_xticks(labels)

    for i in range(len(labels)):
        if i != 0 and i != len(labels) - 1:
            value = "{:,.1f}".format(-heights[i])
        else:
            value = "{:,.1f}".format(heights[i])
        y_pos = white_bar_heights[i] + np.max([heights[i], 0])
        ax.text(labels[i], y_pos, value, ha='center', va='bottom', zorder=50)

    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=15, break_long_words=False))
    ax.set_xticklabels(labels)
    
    # Calculate automatic y-axis limits based on LCOE values
    min_lcoe = min(total_df1, total_df2)
    max_lcoe = max(total_df1, total_df2)
    
    # Add padding for better visualization
    padding_below = 2.0
    # Increase padding above when y_min = 0 to accommodate text labels
    padding_above = 1.0  # Increased to provide space for text labels above bars
    
    #y_min = 0
    y_min = min_lcoe - padding_below
    y_max = max_lcoe + padding_above
    
    # Round to nice numbers (to nearest 0.5 for cleaner axis labels)
    y_min = np.floor(y_min * 2) / 2
    y_max = np.ceil(y_max * 2) / 2
    
    # Set y-axis limit
    ax.set_ylim([y_min, y_max])
    #plt.grid(axis = 'y', zorder = 100)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


    
    fig.tight_layout()
    #fig.savefig("C:\\Code\\WAVES\\library\\Standardized_Moorings\\results\\comparison_waterfall.png", bbox_inches='tight')
    #fig.savefig("C:\\Code\\WAVES\\library\\Standardized_Moorings3\\results\\comparison_waterfall.png", bbox_inches='tight')
    if outfile:
        fig.savefig(outfile, bbox_inches='tight')

    plt.show()


def load_and_average_project_runs(results_dir_path, num_runs=10):
    """
    Load and average the LCOE breakdown data from multiple project run CSV files.
    
    Parameters:
    -----------
    results_dir_path : Path or str
        Path to the results directory containing the project_run*.csv files
    num_runs : int
        Number of runs to average (default: 10)
        
    Returns:
    --------
    pd.DataFrame
        Averaged LCOE breakdown dataframe with same structure as individual runs
    """
    results_path = Path(results_dir_path)
    
    # Load all project run CSV files
    project_dfs = []
    for i in range(1, num_runs + 1):
        csv_file = results_path / f"project_run{i}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file, index_col=0)
            project_dfs.append(df)
        else:
            print(f"Warning: {csv_file} not found, skipping...")
    
    if not project_dfs:
        raise ValueError(f"No project run CSV files found in {results_path}")
    
    print(f"Loaded {len(project_dfs)} project run files from {results_path}")
    
    # Concatenate all dataframes and calculate mean for each component
    df_concat = pd.concat(project_dfs, ignore_index=False)
    
    # Separate numeric and non-numeric columns
    numeric_cols = df_concat.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df_concat.select_dtypes(exclude=[np.number]).columns
    
    # Calculate mean for numeric columns only
    if len(numeric_cols) > 0:
        df_avg_numeric = df_concat[numeric_cols].groupby(df_concat.index).mean()
    else:
        df_avg_numeric = pd.DataFrame(index=df_concat.index.unique())
    
    # For non-numeric columns, take values from the first dataframe (they should be identical across runs)
    if len(non_numeric_cols) > 0 and len(project_dfs) > 0:
        df_non_numeric = project_dfs[0][non_numeric_cols].copy()
        # Combine numeric averages with non-numeric values
        df_avg = df_non_numeric.join(df_avg_numeric, how='outer')
        # Reorder columns to match original structure if possible
        if len(project_dfs[0].columns) > 0:
            original_col_order = [col for col in project_dfs[0].columns if col in df_avg.columns]
            df_avg = df_avg[original_col_order]
    else:
        df_avg = df_avg_numeric
    
    # Remove rows with NaN values in critical columns
    if 'Value ($/MWh)' in df_avg.columns:
        df_avg = df_avg.dropna(subset=['Value ($/MWh)'])
    
    # Remove rows where Component or Category is NaN
    if 'Component' in df_avg.columns:
        df_avg = df_avg.dropna(subset=['Component'])
    if 'Category' in df_avg.columns:
        df_avg = df_avg.dropna(subset=['Category'])
    
    # Reset index to clean up any remaining NaN index values
    df_avg = df_avg.reset_index(drop=True)
    
    return df_avg


def create_individual_waterfall_charts_from_project_runs(baseline_results_dir, interrow_results_dir, output_dir=None, num_runs=10):
    """
    Create individual waterfall charts from averaged project run data.
    
    Parameters:
    -----------
    baseline_results_dir : Path or str
        Path to baseline results directory containing project_run*.csv files
    interrow_results_dir : Path or str  
        Path to interrow results directory containing project_run*.csv files
    output_dir : Path or str, optional
        Directory to save output charts. If None, uses baseline_results_dir/figures
    num_runs : int
        Number of runs to average (default: 10)
    """
    
    print("Creating individual waterfall charts from averaged project run data...")
    
    # Load and average the project run data
    df_baseline_avg = load_and_average_project_runs(baseline_results_dir, num_runs)
    df_interrow_avg = load_and_average_project_runs(interrow_results_dir, num_runs)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(baseline_results_dir) / "figures"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create individual waterfall charts for averaged data
    baseline_outfile = output_dir / "averaged_individual_waterfall_baseline.png"
    interrow_outfile = output_dir / "averaged_individual_waterfall_interrow.png"
    
    print(f"Creating individual waterfall charts from averaged project runs...")
    
    # Individual waterfall charts
    plot_LCOE_waterfall('Baseline (Averaged Project Runs)', df_baseline_avg, outfile=baseline_outfile)
    plot_LCOE_waterfall('Fishing-Informed (Averaged Project Runs)', df_interrow_avg, outfile=interrow_outfile)
    
    print(f"Individual waterfall charts saved to: {output_dir}")
    
    return df_baseline_avg, df_interrow_avg


def load_lcoe_results_and_create_breakdown(results_dir_path):
    """
    Load LCOE results CSV and convert to waterfall chart format.
    
    Parameters:
    -----------
    results_dir_path : Path or str
        Path to the results directory containing the lcoe_results.csv file
        
    Returns:
    --------
    tuple: (pd.DataFrame, float)
        LCOE breakdown dataframe formatted for waterfall charts, and the reported LCOE value
    """
    results_path = Path(results_dir_path)
    lcoe_file = results_path / "lcoe_results.csv"
    
    if not lcoe_file.exists():
        raise ValueError(f"LCOE results file not found: {lcoe_file}")
    
    # Load the LCOE results
    df_report = pd.read_csv(lcoe_file, index_col=0)
    
    print(f"Loaded LCOE results from: {lcoe_file}")
    print(f"Available metrics: {list(df_report.index)}")
    
    # Extract LCOE components from the report
    # Look for CapEx components (they start with "CapEx - ")
    capex_metrics = df_report[df_report.index.str.contains("CapEx -", na=False)]
    
    # Get OpEx if available
    opex_metrics = df_report[df_report.index.str.contains("OpEx", na=False)]
    
    # Get key values
    try:
        fcr = df_report.loc["FCR (%)", df_report.columns[0]] / 100  # Convert percentage to decimal
        net_aep = df_report.loc["AEP per kW (MWh/kW)", df_report.columns[0]]
        total_lcoe = df_report.loc["LCOE ($/MWh)", df_report.columns[0]]
    except KeyError as e:
        print(f"Warning: Could not find expected metric in LCOE results: {e}")
        # Use default values if metrics are missing
        fcr = 0.075  # Default FCR
        net_aep = 4.0  # Default net AEP
        total_lcoe = df_report.loc[df_report.index.str.contains("LCOE", na=False)].iloc[0, 0] if any(df_report.index.str.contains("LCOE", na=False)) else 100
    
    # Create waterfall format dataframe
    waterfall_data = []
    
    # Process CapEx components
    for idx, row in capex_metrics.iterrows():
        component_name = idx.replace("CapEx - ", "").replace(" ($/kW)", "")
        value_per_kw = row.iloc[0]
        value_per_kw_yr = value_per_kw * fcr
        value_per_mwh = value_per_kw_yr / net_aep
        
        waterfall_data.append({
            "Component": component_name,
            "Category": "CapEx",
            "Value ($/kW)": value_per_kw,
            "Fixed charge rate (FCR) (real)": fcr,
            "Value ($/kW-yr)": value_per_kw_yr,
            "Net AEP (MWh/kW/yr)": net_aep,
            "Value ($/MWh)": value_per_mwh
        })
    
    # Process OpEx components
    for idx, row in opex_metrics.iterrows():
        if "per kW" in idx:
            component_name = idx.replace("OpEx per kW ($/kW)", "OpEx").replace("Annual OpEx per kW ($/kW)", "OpEx")
            value_per_kw_yr = row.iloc[0]  # This is already per year
            value_per_kw = value_per_kw_yr / fcr  # Back-calculate total value
            value_per_mwh = value_per_kw_yr / net_aep
            
            waterfall_data.append({
                "Component": "OpEx",
                "Category": "OpEx", 
                "Value ($/kW)": value_per_kw,
                "Fixed charge rate (FCR) (real)": fcr,
                "Value ($/kW-yr)": value_per_kw_yr,
                "Net AEP (MWh/kW/yr)": net_aep,
                "Value ($/MWh)": value_per_mwh
            })
            break  # Only need one OpEx entry
    
    # Create DataFrame
    df_waterfall = pd.DataFrame(waterfall_data)
    
    # Validate that our calculated LCOE matches the reported LCOE
    calculated_lcoe = df_waterfall["Value ($/MWh)"].sum()
    print(f"Calculated LCOE: {calculated_lcoe:.2f} $/MWh")
    print(f"Reported LCOE: {total_lcoe:.2f} $/MWh")
    print(f"Difference: {abs(calculated_lcoe - total_lcoe):.2f} $/MWh")
    
    return df_waterfall, total_lcoe


def create_comparison_waterfall_from_lcoe_results(baseline_results_dir, interrow_results_dir, output_dir=None):
    """
    Create comparison waterfall chart using the lcoe_results.csv files.
    
    Parameters:
    -----------
    baseline_results_dir : Path or str
        Path to baseline results directory containing lcoe_results.csv
    interrow_results_dir : Path or str  
        Path to interrow results directory containing lcoe_results.csv
    output_dir : Path or str, optional
        Directory to save output charts. If None, uses baseline_results_dir/figures
    """
    
    print("Creating comparison waterfall chart from LCOE results CSV files...")
    
    # Load and convert LCOE results to waterfall format
    df_baseline, baseline_lcoe_reported = load_lcoe_results_and_create_breakdown(baseline_results_dir)
    df_interrow, interrow_lcoe_reported = load_lcoe_results_and_create_breakdown(interrow_results_dir)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(baseline_results_dir) / "figures"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison waterfall chart
    comparison_outfile = output_dir / "lcoe_results_comparison_waterfall.png"
    
    # Comparison waterfall chart only
    create_waterfall_chart(
        df_baseline, 
        'Baseline', 
        df_interrow, 
        'Fishing-Informed', 
        'lightskyblue', 
        outfile=comparison_outfile
    )
    
    return df_baseline, df_interrow








######################################################################################################






run = 0
NUM_RUNS = 10
case = 1
# Case 1: Comparing Baseline Humboldt to Standardized Humboldt for Standardized Moorings project
# Case 2: Comparing Baseline Morro Bay to Standardized Morro Bay for Standardized Moorings project
# Case 3: Comparing Baseline Gulf of Maine to Fishing-Informed Gulf of Maine for Fishing Co-Design project
# Case 4: Comparing Baseline Mid-Atlantic to Fishing-Informed Mid-Atlantic for Fishing Co-Design project
# run=1 runs the WAVES code, run=0 plots the WAVES results

if run:

    if case==1:
        projects_df_humboldt_baseline, report_df_humboldt_baseline = runRunWAVES(NUM_RUNS, Path("library/Standardized_Moorings/"))              # Humboldt Baseline
        if "DATA_LIBRARY" in os.environ:
            del os.environ["DATA_LIBRARY"]
        projects_df_humboldt_standardized, report_df_humboldt_standardized = runRunWAVES(NUM_RUNS, Path("library/Standardized_Moorings2/"))     # Humboldt Standardized
    elif case==2:
        projects_df_morrobay_baseline, report_df_morrobay_baseline = runRunWAVES(NUM_RUNS, Path("library/Standardized_Moorings3/"))             # Morro Bay Baseline
        if "DATA_LIBRARY" in os.environ:
            del os.environ["DATA_LIBRARY"]
        projects_df_morrobay_standardized, report_df_morrobay_standardized = runRunWAVES(NUM_RUNS, Path("library/Standardized_Moorings4/"))     # Morro Bay Standardized

    elif case==3:
        projects_df_fcd_gulfofmaine_baseline, report_df_fcd_gulfofmaine_baseline = runRunWAVES(NUM_RUNS, Path("library/FCD_Baseline/"))
        if "DATA_LIBRARY" in os.environ:
            del os.environ["DATA_LIBRARY"]
        projects_df_fcd_gulfofmaine_interrow, report_df_fcd_gulfofmaine_interrow = runRunWAVES(NUM_RUNS, Path("library/FCD_Interrow/"))
    elif case==4:
        projects_df_fcd_midatlantic_baseline, report_df_fcd_midatlantic_baseline = runRunWAVES(NUM_RUNS, Path("library/FCD_MidAtlantic_Baseline/"))
        if "DATA_LIBRARY" in os.environ:
            del os.environ["DATA_LIBRARY"]
        projects_df_fcd_midatlantic_interrow, report_df_fcd_midatlantic_interrow = runRunWAVES(NUM_RUNS, Path("library/FCD_MidAtlantic_Interrow/"))


    if case==1:
        for i,(df_project1,df_project2) in enumerate(zip(projects_df_humboldt_baseline, projects_df_humboldt_standardized)):
            df_project1.to_csv(Path("library/Standardized_Moorings/results") / f"project_run{i+1}.csv")
            df_project2.to_csv(Path("library/Standardized_Moorings2/results") / f"project_run{i+1}.csv")
            plot_LCOE_waterfall('test', df_project1, outfile=Path("library/Standardized_Moorings/results/figures") / f"LCOE_waterfall{i+1}.png")
            plot_LCOE_waterfall('test', df_project2, outfile=Path("library/Standardized_Moorings2/results/figures") / f"LCOE_waterfall{i+1}.png")
            create_waterfall_chart(df_project1, 'Baseline', df_project2, 'Fishing-Informed', 'lightskyblue', outfile=Path("library/Standardized_Moorings/results/figures") / f"comparison_LCOE_waterfall{i+1}.png")
    elif case==2:
        for i,(df_project1,df_project2) in enumerate(zip(projects_df_morrobay_baseline, projects_df_morrobay_standardized)):
            df_project1.to_csv(Path("library/Standardized_Moorings3/results") / f"project_run{i+1}.csv")
            df_project2.to_csv(Path("library/Standardized_Moorings4/results") / f"project_run{i+1}.csv")
            plot_LCOE_waterfall('test', df_project1, outfile=Path("library/Standardized_Moorings3/results/figures") / f"LCOE_waterfall{i+1}.png")
            plot_LCOE_waterfall('test', df_project2, outfile=Path("library/Standardized_Moorings4/results/figures") / f"LCOE_waterfall{i+1}.png")
            create_waterfall_chart(df_project1, 'Baseline', df_project2, 'Fishing-Informed', 'lightskyblue', outfile=Path("library/Standardized_Moorings3/results/figures") / f"comparison_LCOE_waterfall{i+1}.png")


    elif case==3:
        for i,(df_project1,df_project2) in enumerate(zip(projects_df_fcd_gulfofmaine_baseline, projects_df_fcd_gulfofmaine_interrow)):
            df_project1.to_csv(Path("library/FCD_Baseline/results") / f"project_run{i+1}.csv")
            df_project2.to_csv(Path("library/FCD_Interrow/results") / f"project_run{i+1}.csv")
            plot_LCOE_waterfall('test', df_project1, outfile=Path("library/FCD_Baseline/results/figures") / f"LCOE_waterfall{i+1}.png")
            plot_LCOE_waterfall('test', df_project2, outfile=Path("library/FCD_Interrow/results/figures") / f"LCOE_waterfall{i+1}.png")
            create_waterfall_chart(df_project1, 'Baseline', df_project2, 'Fishing-Informed', 'lightskyblue', outfile=Path("library/FCD_Baseline/results/figures") / f"comparison_LCOE_waterfall{i+1}.png")
    elif case==4:
        for i,(df_project1,df_project2) in enumerate(zip(projects_df_fcd_midatlantic_baseline, projects_df_fcd_midatlantic_interrow)):
            df_project1.to_csv(Path("library/FCD_MidAtlantic_Baseline/results") / f"project_run{i+1}.csv")
            df_project2.to_csv(Path("library/FCD_MidAtlantic_Interrow/results") / f"project_run{i+1}.csv")
            plot_LCOE_waterfall('test', df_project1, outfile=Path("library/FCD_MidAtlantic_Baseline/results/figures") / f"LCOE_waterfall{i+1}.png")
            plot_LCOE_waterfall('test', df_project2, outfile=Path("library/FCD_MidAtlantic_Interrow/results/figures") / f"LCOE_waterfall{i+1}.png")
            create_waterfall_chart(df_project1, 'Baseline', df_project2, 'Fishing-Informed', 'lightskyblue', outfile=Path("library/FCD_MidAtlantic_Baseline/results/figures") / f"comparison_LCOE_waterfall{i+1}.png")


    if case==1:
        print(report_df_humboldt_baseline)
        print(report_df_humboldt_standardized)
    elif case==2:
        print(report_df_morrobay_baseline)
        print(report_df_morrobay_standardized)

    elif case==3:
        print(report_df_fcd_gulfofmaine_baseline)
        print(report_df_fcd_gulfofmaine_interrow)
    elif case==4:
        print(report_df_fcd_midatlantic_baseline)
        print(report_df_fcd_midatlantic_interrow)





else:
    if case==1:
        print("\n=== Creating Individual Waterfall Charts from Project Runs ===")
        df_baseline_avg, df_interrow_avg = create_individual_waterfall_charts_from_project_runs(
            baseline_results_dir="library/Standardized_Moorings/results",
            interrow_results_dir="library/Standardized_Moorings2/results",
            num_runs=NUM_RUNS
        )
        
        print("\n=== Creating Comparison Waterfall Chart from LCOE Results ===")
        df_baseline_lcoe, df_interrow_lcoe = create_comparison_waterfall_from_lcoe_results(
            baseline_results_dir="library/Standardized_Moorings/results",
            interrow_results_dir="library/Standardized_Moorings2/results"
        )
    elif case==2:
        print("\n=== Creating Individual Waterfall Charts from Project Runs ===")
        df_baseline_avg, df_interrow_avg = create_individual_waterfall_charts_from_project_runs(
            baseline_results_dir="library/Standardized_Moorings3/results",
            interrow_results_dir="library/Standardized_Moorings4/results",
            num_runs=NUM_RUNS
        )
        
        print("\n=== Creating Comparison Waterfall Chart from LCOE Results ===")
        df_baseline_lcoe, df_interrow_lcoe = create_comparison_waterfall_from_lcoe_results(
            baseline_results_dir="library/Standardized_Moorings3/results",
            interrow_results_dir="library/Standardized_Moorings4/results"
        )
    elif case==3:
        print("\n=== Creating Individual Waterfall Charts from Project Runs ===")
        df_baseline_avg, df_interrow_avg = create_individual_waterfall_charts_from_project_runs(
            baseline_results_dir="library/FCD_Baseline/results",
            interrow_results_dir="library/FCD_Interrow/results",
            num_runs=NUM_RUNS
        )
        
        print("\n=== Creating Comparison Waterfall Chart from LCOE Results ===")
        df_baseline_lcoe, df_interrow_lcoe = create_comparison_waterfall_from_lcoe_results(
            baseline_results_dir="library/FCD_Baseline/results",
            interrow_results_dir="library/FCD_Interrow/results"
        )
    elif case==4:
        print("\n=== Creating Individual Waterfall Charts from Project Runs ===")
        df_baseline_avg, df_interrow_avg = create_individual_waterfall_charts_from_project_runs(
            baseline_results_dir="library/FCD_MidAtlantic_Baseline/results",
            interrow_results_dir="library/FCD_MidAtlantic_Interrow/results",
            num_runs=NUM_RUNS
        )
        
        print("\n=== Creating Comparison Waterfall Chart from LCOE Results ===")
        df_baseline_lcoe, df_interrow_lcoe = create_comparison_waterfall_from_lcoe_results(
            baseline_results_dir="library/FCD_MidAtlantic_Baseline/results",
            interrow_results_dir="library/FCD_MidAtlantic_Interrow/results"
        )




a = 2