from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Set, Tuple
import pandas as pd
import io
from datetime import datetime
from collections import defaultdict, Counter
import json
import ast

app = FastAPI(
    title="Healthcare Encounter Analysis API",
    description="API for analyzing missing encounters between EHR and imported data",
    version="1.0.0"
)

class EncounterKey(BaseModel):
    patient_name: str
    provider_name: str
    date_of_service: str

class MissingEncounter(BaseModel):
    encounter_key: EncounterKey
    cpt_codes: List[str]
    total_procedures: int

class AnalysisResult(BaseModel):
    missing_encounters: List[MissingEncounter]
    total_missing: int
    analysis_summary: Dict

class AnalysisInsights(BaseModel):
    patterns: Dict
    provider_analysis: Dict
    date_analysis: Dict
    cpt_analysis: Dict
    recommendations: List[str]

ehr_data = None
imported_data = None

def parse_cpt_codes(cpt_string):
    """Parse CPT codes from string format to list"""
    if pd.isna(cpt_string):
        return []
    
    if isinstance(cpt_string, str):
        if cpt_string.startswith('{') and cpt_string.endswith('}'):
            codes = cpt_string.strip('{}').split(',')
            return [code.strip().strip('"\'') for code in codes if code.strip()]
        elif ',' in cpt_string:
            return [code.strip() for code in cpt_string.split(',')]
        else:
            return [cpt_string.strip()]
    
    return [str(cpt_string)]

def normalize_encounter_data(df, date_col='from_date_range'):
    """Normalize encounter data for comparison"""
    normalized_encounters = []
    
    for _, row in df.iterrows():
        patient_name = str(row['Patient Name']).strip() if pd.notna(row['Patient Name']) else ""
        provider_name = str(row['Provider Name']).strip() if pd.notna(row['Provider Name']) else ""
        
        date_val = row[date_col]
        if pd.isna(date_val):
            formatted_date = ""
        else:
            date_str = str(date_val).strip()
            
            try:
                if '/' in date_str:
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                else:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                formatted_date = date_obj.strftime('%Y-%m-%d')
            except:
                formatted_date = date_str

        if 'cpt_codes' in row and pd.notna(row['cpt_codes']):
            cpt_codes = parse_cpt_codes(row['cpt_codes'])
        elif 'CPT Code' in row and pd.notna(row['CPT Code']):
            cpt_codes = [str(row['CPT Code']).strip()]
        else:
            cpt_codes = []

        if not patient_name or not provider_name or not formatted_date:
            continue
            
        encounter_key = f"{patient_name}|{provider_name}|{formatted_date}"
        
        normalized_encounters.append({
            'encounter_key': encounter_key,
            'patient_name': patient_name,
            'provider_name': provider_name,
            'date_of_service': formatted_date,
            'cpt_codes': cpt_codes
        })
    
    return normalized_encounters

def group_encounters_by_key(encounters):
    """Group encounters by unique key and combine CPT codes"""
    grouped = defaultdict(lambda: {
        'patient_name': '',
        'provider_name': '',
        'date_of_service': '',
        'cpt_codes': set()
    })
    
    for encounter in encounters:
        key = encounter['encounter_key']
        grouped[key]['patient_name'] = encounter['patient_name']
        grouped[key]['provider_name'] = encounter['provider_name']
        grouped[key]['date_of_service'] = encounter['date_of_service']
        grouped[key]['cpt_codes'].update(encounter['cpt_codes'])

    for key in grouped:
        grouped[key]['cpt_codes'] = list(grouped[key]['cpt_codes'])
    
    return grouped

@app.post("/upload-ehr-data")
async def upload_ehr_data(file: UploadFile = File(...)):
    """Upload EHR closed encounters data"""
    global ehr_data
    
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))

        required_columns = ['Patient Name', 'Provider Name', 'Date of Service', 'CPT Code']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns. Expected: {required_columns}"
            )
        
        ehr_data = df
        return {"message": f"EHR data uploaded successfully. {len(df)} records processed."}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing EHR data: {str(e)}")

@app.post("/upload-imported-data")
async def upload_imported_data(file: UploadFile = File(...)):
    """Upload imported encounters data"""
    global imported_data
    
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        required_columns = ['Patient Name', 'Provider Name', 'from_date_range', 'cpt_codes']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns. Expected: {required_columns}"
            )
        
        imported_data = df
        return {"message": f"Imported data uploaded successfully. {len(df)} records processed."}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing imported data: {str(e)}")

@app.get("/find-missing-encounters", response_model=AnalysisResult)
async def find_missing_encounters():
    """Task 1: Find encounters missing from imported data"""
    global ehr_data, imported_data
    
    if ehr_data is None or imported_data is None:
        raise HTTPException(
            status_code=400, 
            detail="Both EHR and imported data must be uploaded first"
        )
    
    try:
        ehr_encounters = normalize_encounter_data(ehr_data, 'Date of Service')
        imported_encounters = normalize_encounter_data(imported_data, 'from_date_range')
        
        ehr_grouped = group_encounters_by_key(ehr_encounters)
        imported_grouped = group_encounters_by_key(imported_encounters)
        
        missing_encounters = []
        ehr_keys = set(ehr_grouped.keys())
        imported_keys = set(imported_grouped.keys())
        missing_keys = ehr_keys - imported_keys
        
        for key in missing_keys:
            encounter_data = ehr_grouped[key]
            missing_encounter = MissingEncounter(
                encounter_key=EncounterKey(
                    patient_name=encounter_data['patient_name'],
                    provider_name=encounter_data['provider_name'],
                    date_of_service=encounter_data['date_of_service']
                ),
                cpt_codes=encounter_data['cpt_codes'],
                total_procedures=len(encounter_data['cpt_codes'])
            )
            missing_encounters.append(missing_encounter)
        
        analysis_summary = {
            "total_ehr_encounters": len(ehr_grouped),
            "total_imported_encounters": len(imported_grouped),
            "missing_count": len(missing_encounters),
            "import_success_rate": round((len(imported_grouped) / len(ehr_grouped)) * 100, 2) if len(ehr_grouped) > 0 else 0
        }
        
        return AnalysisResult(
            missing_encounters=missing_encounters,
            total_missing=len(missing_encounters),
            analysis_summary=analysis_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing encounters: {str(e)}")

@app.get("/analyze-missing-patterns", response_model=AnalysisInsights)
async def analyze_missing_patterns():
    """Task 2: Deep analysis of why encounters are missing"""
    global ehr_data, imported_data
    
    if ehr_data is None or imported_data is None:
        raise HTTPException(
            status_code=400, 
            detail="Both EHR and imported data must be uploaded first"
        )
    
    try:
        missing_result = await find_missing_encounters()
        missing_encounters = missing_result.missing_encounters
        
        if not missing_encounters:
            return AnalysisInsights(
                patterns={},
                provider_analysis={},
                date_analysis={},
                cpt_analysis={},
                recommendations=["No missing encounters found - all EHR data successfully imported!"]
            )
        
        missing_providers = [enc.encounter_key.provider_name for enc in missing_encounters]
        missing_dates = [enc.encounter_key.date_of_service for enc in missing_encounters]
        missing_cpt_codes = []
        for enc in missing_encounters:
            missing_cpt_codes.extend(enc.cpt_codes)
        
        provider_counts = Counter(missing_providers)
        total_providers_in_ehr = ehr_data['Provider Name'].nunique()
        providers_with_missing = len(provider_counts)
        
        missing_date_objects = []
        for date_str in missing_dates:
            try:
                missing_date_objects.append(datetime.strptime(date_str, '%Y-%m-%d'))
            except:
                continue
        
        month_counts = Counter([d.strftime('%Y-%m') for d in missing_date_objects])
        weekday_counts = Counter([d.strftime('%A') for d in missing_date_objects])
        
        cpt_counts = Counter(missing_cpt_codes)
        unique_cpt_missing = len(set(missing_cpt_codes))

        procedure_counts = [enc.total_procedures for enc in missing_encounters]
        avg_procedures_missing = sum(procedure_counts) / len(procedure_counts) if procedure_counts else 0

        imported_encounters = normalize_encounter_data(imported_data, 'from_date_range')
        imported_providers = [enc['provider_name'] for enc in imported_encounters]
        imported_provider_counts = Counter(imported_providers)

        provider_success_rates = {}
        for provider in set(list(provider_counts.keys()) + list(imported_provider_counts.keys())):
            missing_count = provider_counts.get(provider, 0)
            imported_count = imported_provider_counts.get(provider, 0)
            total_count = missing_count + imported_count
            if total_count > 0:
                success_rate = (imported_count / total_count) * 100
                provider_success_rates[provider] = {
                    'success_rate': round(success_rate, 2),
                    'missing_count': missing_count,
                    'imported_count': imported_count,
                    'total_count': total_count
                }
        
        recommendations = []
        
        worst_providers = sorted(provider_success_rates.items(), key=lambda x: x[1]['success_rate'])[:3]
        if worst_providers and worst_providers[0][1]['success_rate'] < 50:
            providers_list = [p[0] for p in worst_providers if p[1]['success_rate'] < 50]
            recommendations.append(f"Critical: Providers with low import success rates need investigation: {', '.join(providers_list)}")

        most_problematic_cpts = cpt_counts.most_common(3)
        if most_problematic_cpts:
            cpt_list = [f"{cpt} ({count} times)" for cpt, count in most_problematic_cpts]
            recommendations.append(f"Focus on CPT codes with highest failure rates: {', '.join(cpt_list)}")
        
        if month_counts:
            most_problematic_month = month_counts.most_common(1)[0]
            recommendations.append(f"Month {most_problematic_month[0]} has the highest missing encounter count ({most_problematic_month[1]} encounters)")

        if avg_procedures_missing > 3:
            recommendations.append(f"High-complexity encounters (avg {avg_procedures_missing:.1f} procedures) are failing to import - check batch processing limits")

        if len(missing_encounters) > len(imported_encounters) * 0.1:  # More than 10% missing
            recommendations.append("High missing encounter rate suggests systematic import issues - review data validation rules")
        
        if not recommendations:
            recommendations.append("Import success rate is good - monitor for any emerging patterns")
        
        return AnalysisInsights(
            patterns={
                "total_missing_encounters": len(missing_encounters),
                "most_affected_providers": dict(provider_counts.most_common(5)),
                "missing_by_month": dict(month_counts.most_common()),
                "missing_by_weekday": dict(weekday_counts.most_common()),
                "average_procedures_per_missing_encounter": round(avg_procedures_missing, 2)
            },
            provider_analysis={
                "total_providers_in_system": total_providers_in_ehr,
                "providers_with_missing_encounters": providers_with_missing,
                "provider_success_rates": dict(sorted(provider_success_rates.items(), 
                                                    key=lambda x: x[1]['success_rate'])),
                "worst_performing_providers": [p[0] for p in worst_providers][:5]
            },
            date_analysis={
                "date_range_analysis": {
                    "earliest_missing": min(missing_dates) if missing_dates else None,
                    "latest_missing": max(missing_dates) if missing_dates else None
                },
                "temporal_patterns": dict(month_counts.most_common()),
                "weekday_distribution": dict(weekday_counts.most_common())
            },
            cpt_analysis={
                "unique_cpt_codes_missing": unique_cpt_missing,
                "most_problematic_cpt_codes": dict(cpt_counts.most_common(10)),
                "total_procedure_instances_missing": len(missing_cpt_codes)
            },
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing patterns: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ehr_data_loaded": ehr_data is not None,
        "imported_data_loaded": imported_data is not None
    }

@app.get("/data-summary")
async def get_data_summary():
    """Get summary of loaded data"""
    global ehr_data, imported_data
    
    summary = {
        "ehr_data": None,
        "imported_data": None
    }
    
    if ehr_data is not None:
        ehr_dates = pd.to_datetime(ehr_data['Date of Service'], errors='coerce').dropna()
        
        summary["ehr_data"] = {
            "total_records": len(ehr_data),
            "unique_patients": ehr_data['Patient Name'].nunique(),
            "unique_providers": ehr_data['Provider Name'].nunique(),
            "date_range": {
                "earliest": ehr_dates.min().strftime('%Y-%m-%d') if len(ehr_dates) > 0 else None,
                "latest": ehr_dates.max().strftime('%Y-%m-%d') if len(ehr_dates) > 0 else None
            },
            "columns": list(ehr_data.columns)
        }
    
    if imported_data is not None:
        imported_dates = pd.to_datetime(imported_data['from_date_range'], errors='coerce').dropna()
        
        summary["imported_data"] = {
            "total_records": len(imported_data),
            "unique_patients": imported_data['Patient Name'].nunique(),
            "unique_providers": imported_data['Provider Name'].nunique(),
            "date_range": {
                "earliest": imported_dates.min().strftime('%Y-%m-%d') if len(imported_dates) > 0 else None,
                "latest": imported_dates.max().strftime('%Y-%m-%d') if len(imported_dates) > 0 else None
            },
            "columns": list(imported_data.columns)
        }
    
    return summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
