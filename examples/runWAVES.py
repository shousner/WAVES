from copy import deepcopy
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from waves import Project
from waves.utilities import load_yaml

# Update core Pandas display settings
pd.options.display.float_format = "{:,.2f}".format
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

NUM_RUNS = 1
rng = np.random.default_rng(seed=834)

# Containers for each metric
capex_dfs = []
opex_dfs = []
charter_days_dfs = []
mobilization_dfs = []
delay_dfs = []
failure_cost_dfs = []
equipment_cost_dfs = []
report_dfs = []
losses_dfs = []

library_path = Path("library/Standardized_Moorings/")
config_floating = load_yaml(library_path / "project/config", "base_floating_deep.yaml")
config_floating["floris_config"] = load_yaml(library_path / "project/config", config_floating["floris_config"])
config_floating["floris_config"]["farm"]["turbine_library_path"] = library_path / "turbines"
config_floating.update({"library_path": library_path})

config_wombat = load_yaml(library_path / "project/config", config_floating["wombat_config"])

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

        #8. Losses report
        report_df_losses = project_floating.loss_ratio(breakdown=True)
        losses_dfs.append(report_df_losses) 

    return project_floating


'''
# === Compute and save averages ===
__location__ = "C:\\Code\\WAVES\\examples\\standmoor-results\\"
average_and_save(capex_dfs, __location__+"deep_average_capex.csv", index_cols="Component")
average_and_save(opex_dfs, __location__+"deep_average_opex.csv")
average_and_save(charter_days_dfs, __location__+"deep_average_charter_days.csv", index_cols="vessel")
average_and_save(mobilization_dfs, __location__+"deep_average_mobilization_summary.csv", index_cols="agent")
average_and_save(delay_dfs, __location__+"deep_average_delay_summary.csv", index_cols=["agent", "additional"])
average_and_save(failure_cost_dfs, __location__+"deep_average_failures_costs.csv", index_cols=["subassembly", "task"])
average_and_save(equipment_cost_dfs, __location__+"deep_average_equipment_costs.csv")
average_and_save(report_dfs, __location__+"deep_average_report_df.csv", index_cols="Metrics")
average_and_save(losses_dfs, __location__+"deep_average_losses_report_df.csv")
'''



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



def plot_LCOE_waterfall(technology, df, width=8, height=6, y_min=None, y_max=None):
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
    fig.savefig("C:\\Code\\WAVES\\library\\Standardized_Moorings3\\results\\LCOE_waterfall.png", bbox_inches='tight')
    plt.show()



def create_waterfall_chart(df_1, total_1, df_2, total_2, color_2):
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

    colors = ['red' if h < 0 else 'green' for h in heights]
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
    
    # Set y-axis limit
    ax.set_ylim([115, 121])
    #ax.set_ylim([128, 133.5])
    #plt.grid(axis = 'y', zorder = 100)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


    
    fig.tight_layout()
    #fig.savefig("C:\\Code\\WAVES\\library\\Standardized_Moorings\\results\\comparison_waterfall.png", bbox_inches='tight')
    #fig.savefig("C:\\Code\\WAVES\\library\\Standardized_Moorings3\\results\\comparison_waterfall.png", bbox_inches='tight')

    plt.show()



# Standardized_Moorings = Humboldt Baseline
# Standardized_Moorings2 = Humboldt Standardized
# Standardized_Moorings3 = Morro Bay Baseline
# Standardized_Moorings4 = Morro Bay Standardized

'''
project_floating1 = runRunWAVES(10, Path("library/Standardized_Moorings/"))
project_floating2 = runRunWAVES(10, Path("library/Standardized_Moorings2/"))

project_floating1 = runRunWAVES(10, Path("library/Standardized_Moorings3/"))
project_floating2 = runRunWAVES(10, Path("library/Standardized_Moorings4/"))
'''

#project_floating_fcd_gulfofmaine_baseline = runRunWAVES(10, Path("library/FCD_Baseline/"))
#project_floating_fcd_gulfofmaine_interrow = runRunWAVES(10, Path("library/FCD_Interrow/"))
#project_floating_fcd_midatlantic_baseline = runRunWAVES(1, Path("library/FCD_MidAtlantic_Baseline/"))
project_floating_fcd_midatlantic_interrow = runRunWAVES(1, Path("library/FCD_MidAtlantic_Interrow/"))

#df1 = project_floating1.generate_report_lcoe_breakdown()
#df2 = project_floating2.generate_report_lcoe_breakdown()
#df1 = generate_report_lcoe_breakdown_adjusted(project_floating1)
#df2 = generate_report_lcoe_breakdown_adjusted(project_floating2)

#df3 = generate_report_lcoe_breakdown_adjusted(project_floating_fcd_gulfofmaine_baseline)
#df4 = generate_report_lcoe_breakdown_adjusted(project_floating_fcd_gulfofmaine_interrow)

#df5 = generate_report_lcoe_breakdown_adjusted(project_floating_fcd_midatlantic_baseline)
df6 = generate_report_lcoe_breakdown_adjusted(project_floating_fcd_midatlantic_interrow)

#plot_LCOE_waterfall('test', df3)
#plot_LCOE_waterfall('test', df4)
#plot_LCOE_waterfall('test', df5)
plot_LCOE_waterfall('test', df6)


#create_waterfall_chart(df1, 'Baseline', df2, 'Standardized', 'lightskyblue')
create_waterfall_chart(df3, 'Baseline', df4, 'Fishing-Informed', 'lightskyblue')
#create_waterfall_chart(df5, 'Baseline', df6, 'Fishing-Informed', 'lightskyblue')


'''
df2 = df1.copy()
df2.loc[df2["Component"]=="Mooring System Installation", "Value ($/MWh)"] *= 2
df2.loc[df2["Component"]=="Array System Installation", "Value ($/MWh)"] /= 2
'''






for name,value in project_floating1.orbit.capex_breakdown.items():
    print(name, value/600000)

for name,value in project_floating2.orbit.capex_breakdown.items():
    print(name, value/600000)

capex1 = project_floating1.orbit.total_capex
capex2 = project_floating2.orbit.total_capex

aep_per_kw1 = project_floating1.energy_production(units="mw", per_capacity="kw", aep=True)
aep_per_kw2 = project_floating2.energy_production(units="mw", per_capacity="kw", aep=True)

print(capex1)
print(capex2)
print(aep_per_kw1)
print(aep_per_kw2)

opex_df1 = project_floating1.wombat.metrics.opex(frequency='annual', by_category=True) / project_floating1.capacity("kw")
average_opex1 = opex_df1["OpEx"].mean()

opex_df2 = project_floating2.wombat.metrics.opex(frequency='annual', by_category=True) / project_floating2.capacity("kw")
average_opex2 = opex_df2["OpEx"].mean()

print(average_opex1)
print(average_opex2)

a = 2