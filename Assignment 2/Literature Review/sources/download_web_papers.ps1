# Download papers from websites (non-arXiv)
# Some may require manual browser print-to-PDF

$ErrorActionPreference = "Continue"
$baseDir = Get-Location

# Papers with direct website links
$webDownloads = @(
    # ResearchGate
    @{url="https://www.researchgate.net/figure/BIG-Bench-Hard-1827-individual-task-performance_tbl3_372989811"; file="bucket_a_faithfulness/tier_3_reference/BIG_Bench_Hard_2023_Performance_Data.pdf"; note="ResearchGate figure - may need manual download"},
    @{url="https://www.researchgate.net/scientific-contributions/Jerry-Wei-2152948001"; file="bucket_b_sycophancy/tier_3_reference/Jerry_Wei_2023_Research_Profile.pdf"; note="ResearchGate profile - may need manual download"},
    @{url="https://www.researchgate.net/publication/384212923"; file="bucket_a_faithfulness/tier_3_reference/ResearchGate_2024_Probabilities_Also_Matter.pdf"; note="ResearchGate publication - may need login"},
    
    # Blog articles (HTML - will download as HTML, user can print to PDF)
    @{url="https://www.prompthub.us/blog/why-llms-fail-in-multi-turn-conversations-and-how-to-fix-it"; file="bucket_d_longitudinal/tier_1_must_read/Zheng_2024_Why_LLMs_Fail_Multi_Turn.html"; note="Blog article - use browser print-to-PDF"},
    @{url="https://www.evidentlyai.com/blog/llm-safety-bias-benchmarks"; file="bucket_c_silent_bias/tier_3_reference/Evidently_AI_2024_Safety_Bias_Benchmarks.html"; note="Blog article - use browser print-to-PDF"},
    
    # Documentation
    @{url="https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness"; file="evaluation_tools/tier_3_reference/Ragas_2025_Faithfulness_Documentation.html"; note="Documentation - use browser print-to-PDF"},
    @{url="https://huggingface.co/papers?q=mental%20health%20counseling"; file="clinical_domain/tier_3_reference/HuggingFace_2024_Mental_Health_Papers.html"; note="Search results - use browser print-to-PDF"},
    
    # GitHub (README)
    @{url="https://raw.githubusercontent.com/google/sycophancy-intervention/main/README.md"; file="bucket_b_sycophancy/tier_3_reference/Google_2024_Sycophancy_Intervention_README.md"; note="GitHub README - can convert to PDF"}
)

Write-Host "Downloading web-based papers..." -ForegroundColor Cyan
Write-Host "Note: Some files will be HTML - use browser print-to-PDF to convert" -ForegroundColor Yellow
Write-Host ""

$successCount = 0
$failCount = 0

foreach ($d in $webDownloads) {
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
        
        # Use browser-like headers
        $headers = @{
            "User-Agent" = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            "Accept" = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        
        Invoke-WebRequest -Uri $d.url -OutFile $filePath -Headers $headers -ErrorAction Stop -TimeoutSec 30
        
        if (Test-Path $filePath -PathType Leaf) {
            $fileSize = (Get-Item $filePath).Length / 1KB
            Write-Host " OK ($([math]::Round($fileSize, 1)) KB)" -ForegroundColor Green
            Write-Host "  Note: $($d.note)" -ForegroundColor Gray
            $successCount++
        } else {
            Write-Host " FAILED - File not created" -ForegroundColor Red
            $failCount++
        }
    } catch {
        Write-Host " FAILED - $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "  Note: $($d.note)" -ForegroundColor Gray
        $failCount++
    }
    
    Start-Sleep -Milliseconds 1000
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
Write-Host "For HTML files, open in browser and use Print > Save as PDF" -ForegroundColor Yellow
Write-Host "For ResearchGate papers, you may need to log in and download manually" -ForegroundColor Yellow

