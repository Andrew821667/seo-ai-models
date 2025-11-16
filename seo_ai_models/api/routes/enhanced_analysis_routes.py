"""
Enhanced analysis routes with real-time progress tracking.

Integrates EnhancedSEOAdvisor with WebSocket progress updates.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session
import secrets

from ..auth.models import User, Permission
from ..auth.dependencies import get_current_user, require_permission
from ..infrastructure.database import get_db
from ..websocket.manager import progress_tracker
from ...models.enhanced_advisor import EnhancedSEOAdvisor
from ...autofix.engine import FixComplexity


router = APIRouter(prefix="/enhanced-analysis", tags=["enhanced-analysis"])


# Request/Response models
class EnhancedAnalysisRequest(BaseModel):
    """Request for enhanced analysis."""
    url: HttpUrl
    content: str
    keywords: List[str]
    auto_fix: bool = True
    fix_complexity_limit: str = "simple"  # trivial, simple, moderate, complex, critical


class EnhancedAnalysisResponse(BaseModel):
    """Response for enhanced analysis."""
    analysis_id: str
    status: str
    message: str


class AnalysisStatusResponse(BaseModel):
    """Analysis status response."""
    analysis_id: str
    status: str
    progress: int
    current_step: str
    url: str
    started_at: str
    completed_at: Optional[str] = None
    results: Optional[dict] = None


# Background task for running analysis
async def run_enhanced_analysis_task(
    analysis_id: str,
    user_id: str,
    url: str,
    content: str,
    keywords: List[str],
    auto_fix: bool,
    complexity_limit: FixComplexity
):
    """Background task to run enhanced analysis with progress tracking."""
    try:
        # Start tracking
        await progress_tracker.start_analysis(analysis_id, user_id, url)

        # Initialize Enhanced Advisor
        advisor = EnhancedSEOAdvisor(auto_execute=auto_fix)

        # Step 1: Base SEO Analysis (20%)
        await progress_tracker.update_progress(
            analysis_id,
            20,
            "Running base SEO analysis",
            {"detail": "Analyzing content quality and keywords"}
        )

        # Step 2: Visual Analysis (40%)
        await progress_tracker.update_progress(
            analysis_id,
            40,
            "Analyzing visual content",
            {"detail": "Checking images, alt tags, optimization"}
        )

        # Step 3: Mobile & Core Web Vitals (60%)
        await progress_tracker.update_progress(
            analysis_id,
            60,
            "Checking mobile friendliness",
            {"detail": "Analyzing Core Web Vitals (LCP, FID, CLS)"}
        )

        # Step 4: Running Auto-Fixes (80%)
        if auto_fix:
            await progress_tracker.update_progress(
                analysis_id,
                80,
                "Applying automatic fixes",
                {"detail": f"Auto-fixing issues up to {complexity_limit.value} complexity"}
            )

        # Execute full analysis
        result = advisor.analyze_and_fix(
            url=str(url),
            content=content,
            keywords=keywords,
            auto_fix=auto_fix,
            fix_complexity_limit=complexity_limit
        )

        # Complete
        await progress_tracker.complete_analysis(analysis_id, result)

    except Exception as e:
        await progress_tracker.fail_analysis(analysis_id, str(e))


@router.post("/analyze", response_model=EnhancedAnalysisResponse)
async def start_enhanced_analysis(
    request: EnhancedAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.RUN_ANALYSIS)),
    db: Session = Depends(get_db)
):
    """
    Start enhanced SEO analysis with auto-fix.

    This endpoint:
    1. Starts analysis in background
    2. Returns analysis_id immediately
    3. Client can track progress via WebSocket

    Permissions: RUN_ANALYSIS required
    """
    # Generate analysis ID
    analysis_id = secrets.token_urlsafe(16)

    # Parse complexity limit
    complexity_map = {
        "trivial": FixComplexity.TRIVIAL,
        "simple": FixComplexity.SIMPLE,
        "moderate": FixComplexity.MODERATE,
        "complex": FixComplexity.COMPLEX,
        "critical": FixComplexity.CRITICAL
    }

    complexity_limit = complexity_map.get(
        request.fix_complexity_limit.lower(),
        FixComplexity.SIMPLE
    )

    # Start background task
    background_tasks.add_task(
        run_enhanced_analysis_task,
        analysis_id=analysis_id,
        user_id=current_user.id,
        url=str(request.url),
        content=request.content,
        keywords=request.keywords,
        auto_fix=request.auto_fix,
        complexity_limit=complexity_limit
    )

    return EnhancedAnalysisResponse(
        analysis_id=analysis_id,
        status="started",
        message=f"Analysis started. Connect to WebSocket to track progress: /ws/analysis/{analysis_id}"
    )


@router.get("/status/{analysis_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(
    analysis_id: str,
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYSIS))
):
    """
    Get current status of analysis.

    Permissions: VIEW_ANALYSIS required
    """
    if analysis_id not in progress_tracker.active_analyses:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found or completed"
        )

    analysis = progress_tracker.active_analyses[analysis_id]

    return AnalysisStatusResponse(
        analysis_id=analysis["id"],
        status=analysis["status"],
        progress=analysis["progress"],
        current_step=analysis["current_step"],
        url=analysis["url"],
        started_at=analysis["started_at"],
        completed_at=analysis.get("completed_at"),
        results=analysis.get("results")
    )


@router.get("/active", response_model=List[AnalysisStatusResponse])
async def get_active_analyses(
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYSIS))
):
    """
    Get all active analyses.

    Permissions: VIEW_ANALYSIS required
    """
    analyses = progress_tracker.get_active_analyses()

    return [
        AnalysisStatusResponse(
            analysis_id=a["id"],
            status=a["status"],
            progress=a["progress"],
            current_step=a["current_step"],
            url=a["url"],
            started_at=a["started_at"],
            completed_at=a.get("completed_at"),
            results=a.get("results")
        )
        for a in analyses
    ]


# AutoFix specific endpoints
class AutoFixApprovalRequest(BaseModel):
    """Request to approve/reject auto-fix."""
    action_id: str
    approved: bool
    reason: Optional[str] = None


@router.post("/autofix/approve")
async def approve_autofix(
    request: AutoFixApprovalRequest,
    current_user: User = Depends(require_permission(Permission.APPROVE_AUTOFIX))
):
    """
    Approve or reject pending auto-fix action.

    Permissions: APPROVE_AUTOFIX required
    """
    # Implementation would interact with AutoFix Engine
    # to approve/reject pending actions

    return {
        "action_id": request.action_id,
        "approved": request.approved,
        "approved_by": current_user.username,
        "message": "Auto-fix action processed"
    }


@router.get("/autofix/pending")
async def get_pending_autofixes(
    current_user: User = Depends(require_permission(Permission.APPROVE_AUTOFIX))
):
    """
    Get all pending auto-fix actions awaiting approval.

    Permissions: APPROVE_AUTOFIX required
    """
    # This would query AutoFix Engine for pending approvals
    return {
        "pending_actions": [],
        "count": 0
    }
