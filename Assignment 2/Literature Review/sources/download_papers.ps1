# Paper Download Script
# Downloads papers from DOWNLOAD_GUIDE.md to appropriate folders

$ErrorActionPreference = "Continue"
$baseDir = Get-Location

# Ensure all tier folders exist
$tierFolders = @(
    "bucket_a_faithfulness/tier_1_must_read",
    "bucket_a_faithfulness/tier_2_important",
    "bucket_a_faithfulness/tier_3_reference",
    "bucket_b_sycophancy/tier_1_must_read",
    "bucket_b_sycophancy/tier_2_important",
    "bucket_b_sycophancy/tier_3_reference",
    "bucket_c_silent_bias/tier_2_important",
    "bucket_d_longitudinal/tier_1_must_read",
    "bucket_d_longitudinal/tier_2_important",
    "bucket_d_longitudinal/tier_3_reference",
    "clinical_domain/tier_1_must_read",
    "clinical_domain/tier_2_important",
    "evaluation_tools/tier_1_must_read",
    "evaluation_tools/tier_2_important",
    "evaluation_tools/tier_3_reference"
)

foreach ($folder in $tierFolders) {
    $fullPath = Join-Path $baseDir $folder
    New-Item -ItemType Directory -Force -Path $fullPath | Out-Null
}

# Papers with direct download links
$downloads = @(
    @{url="https://arxiv.org/pdf/2307.13702.pdf"; file="bucket_a_faithfulness/tier_1_must_read/Lanham_2023_Measuring_Faithfulness.pdf"},
    @{url="https://proceedings.neurips.cc/paper_files/paper/2023/file/ed3fea9033a80fea1376299fa7863f4a-Paper-Conference.pdf"; file="bucket_a_faithfulness/tier_1_must_read/Turpin_2023_Language_Models_Dont_Always_Say.pdf"},
    @{url="https://arxiv.org/pdf/2301.13379.pdf"; file="bucket_a_faithfulness/tier_2_important/Lyu_2023_Faithful_Chain_of_Thought.pdf"},
    @{url="https://arxiv.org/pdf/2502.18848.pdf"; file="bucket_a_faithfulness/tier_2_important/Ibrahim_2025_Causal_Lens_Faithfulness.pdf"},
    @{url="https://arxiv.org/pdf/2510.04040.pdf"; file="bucket_a_faithfulness/tier_2_important/FaithCoT_Bench_2025_Benchmarking_Faithfulness.pdf"},
    @{url="https://arxiv.org/pdf/2407.10114.pdf"; file="bucket_a_faithfulness/tier_3_reference/Tsai_2024_TokenSHAP.pdf"},
    @{url="https://arxiv.org/pdf/2303.17651.pdf"; file="bucket_a_faithfulness/tier_3_reference/Madaan_2023_Self_Refine.pdf"},
    @{url="https://arxiv.org/pdf/2303.11366.pdf"; file="bucket_a_faithfulness/tier_3_reference/Shinn_2023_Reflexion.pdf"},
    @{url="https://openreview.net/pdf?id=YAGa8upUSA"; file="bucket_a_faithfulness/tier_3_reference/BenchRisk_2024_Risk_Management.pdf"},
    @{url="https://arxiv.org/pdf/2308.03958.pdf"; file="bucket_b_sycophancy/tier_1_must_read/Wei_2023_Simple_Synthetic_Data_Reduces_Sycophancy.pdf"},
    @{url="https://arxiv.org/pdf/2310.13548.pdf"; file="bucket_b_sycophancy/tier_1_must_read/Anthropic_2024_Towards_Understanding_Sycophancy.pdf"},
    @{url="https://arxiv.org/pdf/2502.08177.pdf"; file="bucket_b_sycophancy/tier_1_must_read/Fanous_2025_SycEval.pdf"},
    @{url="https://arxiv.org/pdf/2510.16727.pdf"; file="bucket_b_sycophancy/tier_1_must_read/Pandey_2025_Beacon.pdf"},
    @{url="https://arxiv.org/pdf/2503.11656.pdf"; file="bucket_b_sycophancy/tier_1_must_read/Liu_2025_Truth_Decay.pdf"},
    @{url="https://arxiv.org/pdf/2506.21584.pdf"; file="bucket_b_sycophancy/tier_2_important/Koorndijk_2025_Alignment_Faking_Small_LLM.pdf"},
    @{url="https://arxiv.org/pdf/2507.23486.pdf"; file="bucket_c_silent_bias/tier_2_important/Lee_2024_SafeHear.pdf"},
    @{url="https://arxiv.org/pdf/2505.06120.pdf"; file="bucket_d_longitudinal/tier_1_must_read/Laban_2025_LLMs_Get_Lost.pdf"},
    @{url="https://arxiv.org/pdf/2510.07777.pdf"; file="bucket_d_longitudinal/tier_2_important/Yuan_2024_Drift_No_More.pdf"},
    @{url="https://arxiv.org/pdf/2510.05381.pdf"; file="bucket_d_longitudinal/tier_2_important/Context_Length_2025_Performance_Hurt.pdf"},
    @{url="https://arxiv.org/pdf/2505.15715.pdf"; file="clinical_domain/tier_1_must_read/Zhang_2025_Beyond_Empathy_PsyLLM.pdf"},
    @{url="https://arxiv.org/pdf/2507.23486.pdf"; file="clinical_domain/tier_1_must_read/Lee_2024_SafeHear_Clinical.pdf"},
    @{url="https://arxiv.org/pdf/2305.09617.pdf"; file="clinical_domain/tier_2_important/Singhal_2023_Expert_Level_Medical_QA.pdf"},
    @{url="https://arxiv.org/pdf/2507.14079.pdf"; file="clinical_domain/tier_2_important/Chen_2025_DENSE.pdf"},
    @{url="https://arxiv.org/pdf/2501.08977.pdf"; file="evaluation_tools/tier_1_must_read/Kim_2025_PDSQI_9.pdf"},
    @{url="https://www.medrxiv.org/content/10.1101/2025.04.22.25326219v1.full.pdf"; file="evaluation_tools/tier_2_important/Smith_2025_LLM_as_Judge.pdf"}
)

Write-Host "Starting paper downloads..." -ForegroundColor Cyan
Write-Host "Total papers to download: $($downloads.Count)"
Write-Host ""

$successCount = 0
$failCount = 0

foreach ($d in $downloads) {
    $filePath = Join-Path $baseDir $d.file
    $dir = Split-Path $filePath -Parent
    
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    
    if (Test-Path $filePath) {
        Write-Host "Already exists: $($d.file)" -ForegroundColor Green
        $successCount++
        continue
    }
    
    try {
        Write-Host "Downloading: $($d.file)..." -NoNewline
        Invoke-WebRequest -Uri $d.url -OutFile $filePath -ErrorAction Stop -TimeoutSec 30
        
        if (Test-Path $filePath -PathType Leaf) {
            $fileSize = (Get-Item $filePath).Length / 1KB
            Write-Host " OK ($([math]::Round($fileSize, 1)) KB)" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host " FAILED - File not created" -ForegroundColor Red
            $failCount++
        }
    } catch {
        Write-Host " FAILED - $($_.Exception.Message)" -ForegroundColor Red
        $failCount++
    }
    
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "Download Summary:" -ForegroundColor Cyan
Write-Host "  Success: $successCount" -ForegroundColor Green
if ($failCount -eq 0) {
    Write-Host "  Failed:  $failCount" -ForegroundColor Green
} else {
    Write-Host "  Failed:  $failCount" -ForegroundColor Red
}
Write-Host ""
Write-Host "Note: Some papers require manual download (conference proceedings, journals, etc.)" -ForegroundColor Yellow
Write-Host "See DOWNLOAD_GUIDE.md for links to papers that need manual download." -ForegroundColor Yellow
