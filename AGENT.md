# AGENT.md - PV Uncertainty Tool Development Guide

**Project**: Solar PV STC Uncertainty Analysis Tool - Professional Edition  
**Last Updated**: December 15, 2024  
**Version**: 2.0.0-enhanced

## 🎯 Core Principles

### 1. **Safety First - Always Enable Rollback**
- Create feature branches for ALL changes (never commit directly to main)
- Use Alembic migrations for database schema changes (reversible)
- Tag stable releases before major updates
- Maintain backup of production database
- Test rollback procedures before deployment

### 2. **Research-Driven Development**
- Base uncertainty quantification on peer-reviewed papers
- Implement IEC standards accurately (60891, 60904, 61215, 61853)
- Document scientific rationale for each uncertainty component
- Validate against reference implementations

### 3. **Database-First Architecture**
- Design comprehensive schema before implementation
- Use dbdiagram.io for visual schema design
- Normalize data (avoid redundancy)
- Index for query performance
- Enforce referential integrity with foreign keys

### 4. **Modular & Maintainable**
- Separate concerns: ingestion / analysis / database / reporting
- Write docstrings for all functions and classes
- Type hints for improved code clarity
- Unit tests for critical functions
- Keep functions small and focused

### 5. **User-Centric UI**
- Progressive disclosure (basic → advanced features)
- Clear error messages with actionable guidance
- Real-time validation feedback
- Export capabilities (PDF, Excel, JSON)
- Responsive design for different screen sizes

## 🔄 Rollback Procedures

### Database Rollback
```bash
# Check current migration
alembic current

# Rollback one migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade <revision_id>

# Rollback to base (CAUTION: data loss)
alembic downgrade base
```

### Code Rollback
```bash
# Revert last commit (keep changes)
git reset --soft HEAD~1

# Revert last commit (discard changes)
git reset --hard HEAD~1

# Revert to specific commit
git revert <commit_hash>

# Restore specific file from last commit
git checkout HEAD -- <filename>
```

### Railway Deployment Rollback
1. Navigate to Railway project dashboard
2. Click on service → Deployments
3. Find last successful deployment
4. Click "⋮" menu → Redeploy
5. Verify deployment success

### Emergency Rollback Checklist
- [ ] Identify issue and impact scope
- [ ] Check Railway logs for errors
- [ ] Notify stakeholders of rollback
- [ ] Execute database rollback (if needed)
- [ ] Execute code rollback
- [ ] Redeploy previous stable version
- [ ] Verify application functionality
- [ ] Document root cause
- [ ] Create fix branch

## 📊 Enhanced Database Schema (v2.0)

### New Tables (13 total)
1. **organizations** - Lab/manufacturer/research entities
2. **users** - Personnel with roles and permissions
3. **modules** - PV module specifications
4. **measurements** - Test measurement records
5. **temperature_coefficients** - α, β, γ with uncertainties
6. **bifacial_measurements** - φ factors, G_eq calculations
7. **curve_correction_data** - IEC 60891 method tracking
8. **repeatability_statistics** - Type A uncertainty
9. **spectral_responses** - SR measurements
10. **reference_devices** - Calibrated reference cells
11. **spec_simulators** - Simulator specifications
12. **spec_deviations** - Deviation tracking
13. **files** - Document management

### Key Relationships
- organizations → users (1:N)
- organizations → modules (1:N)
- modules → measurements (1:N)
- measurements → temperature_coefficients (1:N)
- measurements → bifacial_measurements (1:1)
- measurements → curve_correction_data (1:1)

## 🛠️ Development Workflow

### 1. Pre-Implementation
```bash
# Update from main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/enhanced-uncertainty-schema

# Check current state
git status
```

### 2. Implementation
```bash
# Make changes in logical commits
git add <files>
git commit -m "feat: add temperature_coefficients table"

# Continue with next feature
git add <files>
git commit -m "feat: add bifacial_measurements table"
```

### 3. Testing
```bash
# Run local tests
pytest tests/

# Check code quality
flake8 src/
black src/ --check

# Test database migrations
alembic upgrade head
alembic downgrade -1
alembic upgrade head
```

### 4. Deployment
```bash
# Push to feature branch
git push origin feature/enhanced-uncertainty-schema

# Create Pull Request on GitHub
# Review changes
# Merge to main after approval

# Railway auto-deploys from main branch
```

## 📝 Current Implementation Status

### Completed ✅
- [x] Basic LIMS-ready schema (Equipment, Project, Sample, Test)
- [x] I-V measurement data storage
- [x] Quality control and deviation tracking
- [x] Streamlit UI with basic uncertainty analysis
- [x] Railway deployment pipeline
- [x] Multi-flasher data ingestion

### In Progress 🔄
- [ ] Enhanced uncertainty schema (13 tables)
- [ ] PostgreSQL integration
- [ ] Alembic migration setup
- [ ] Advanced uncertainty UI components

### Planned 📋
- [ ] Temperature coefficient tracking UI
- [ ] Bifacial measurement module
- [ ] Repeatability statistics dashboard
- [ ] Comprehensive uncertainty budget report
- [ ] ISO 17025 compliant reporting
- [ ] API endpoints for LIMS integration

## 🔍 Quick Reference

### Database Connection
```python
from src.database.connection import get_db_session

with get_db_session() as session:
    # Your queries here
    pass
```

### Adding New Migration
```bash
# Auto-generate migration from schema changes
alembic revision --autogenerate -m "Add temperature_coefficients table"

# Review generated migration file
# Edit if needed

# Apply migration
alembic upgrade head
```

### Railway Environment Variables
```bash
DATABASE_URL=postgresql://...  # Auto-provided by Railway
PYTHON_VERSION=3.13.9
STREAMLIT_SERVER_PORT=8501
```

## 📚 Key Resources

### Standards
- IEC 60904-1: I-V characteristics measurement
- IEC 60891: Temperature and irradiance correction
- IEC 61215: Terrestrial PV modules qualification
- IEC 61853: PV module performance testing
- IEC TS 60904-1-2: Bifacial measurements
- ISO/IEC 17025: Laboratory accreditation
- JCGM 100:2008: GUM (Guide to Expression of Uncertainty)

### Research Papers
- Temperature coefficient uncertainty analysis
- Bifacial module measurement techniques
- Curve correction method comparisons
- Type A/Type B uncertainty propagation

### Tools
- dbdiagram.io: Database schema design
- Alembic: Database migrations
- SQLAlchemy: ORM
- Streamlit: UI framework
- Railway: Deployment platform

## 🎓 Lessons Learned

### Database Design
1. Design schema visually first (dbdiagram.io)
2. Normalize but don't over-normalize
3. Use ENUM types for constrained values
4. Always add created_at/updated_at timestamps
5. Index foreign keys and frequently queried columns

### Deployment
1. Test migrations locally before production
2. Always have rollback plan ready
3. Monitor Railway logs during deployment
4. Use environment variables for configuration
5. Enable automatic deployments only after CI/CD setup

### Code Quality
1. Write docstrings immediately
2. Type hints catch bugs early
3. Small, focused functions are easier to test
4. Separate business logic from UI
5. Document "why" not just "what"

---

**Remember**: Safe, tested, reversible changes. Quality over speed. Research-driven implementation.
