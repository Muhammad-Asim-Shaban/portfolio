'use client';

import { useEffect, useRef } from 'react';
import { GraduationCap, BookOpen, School } from 'lucide-react';

// ─── Data ────────────────────────────────────────────────────────────────────

const educationData = [
  {
    id: 1,
    title: 'Matriculation',
    year: '2021',
    description: 'Completed with a score of 1079/1100 from BISERWP.',
    icon: School,
    side: 'right',
  },
  {
    id: 2,
    title: 'Intermediate',
    year: '2023',
    description: 'Completed with a score of 999/1100.',
    icon: BookOpen,
    side: 'left',
  },
  {
    id: 3,
    title: 'University (CUST)',
    year: 'Present',
    description:
      'Currently pursuing a degree at CUST, with a CGPA of 3.96 after the fifth semester.',
    icon: GraduationCap,
    side: 'right',
  },
];

// ─── Component ───────────────────────────────────────────────────────────────

export default function Qualification() {
  const sectionRef = useRef<HTMLElement>(null);

  // Intersection Observer → fade-in on scroll
  useEffect(() => {
    const cards = sectionRef.current?.querySelectorAll<HTMLElement>('[data-animate]');
    if (!cards) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            (entry.target as HTMLElement).style.opacity = '1';
            (entry.target as HTMLElement).style.transform = 'translateY(0)';
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15 }
    );

    cards.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  return (
    <section
      ref={sectionRef}
      id="qualification"
      className="qualification section"
    >
      {/* ── Section header ─────────────────────────────────────────────── */}
      <div className="container">
        <div className="section__header">
          <p className="section__subtitle">Qualification</p>
          <h2 className="section__title">
            My educational background
            <br />
            and Achievements
          </h2>
        </div>

        {/* ── Timeline ───────────────────────────────────────────────────── */}
        <div className="timeline">
          {/* Vertical line */}
          <div className="timeline__line" aria-hidden="true" />

          {educationData.map((item, index) => {
            const Icon = item.icon;
            const isLeft = item.side === 'left';

            return (
              <div
                key={item.id}
                className={`timeline__item timeline__item--${item.side}`}
                data-animate
                style={{
                  opacity: 0,
                  transform: `translateY(40px)`,
                  transition: `opacity 0.6s ease ${index * 0.18}s, transform 0.6s ease ${index * 0.18}s`,
                }}
              >
                {/* ── Card ─────────────────────────────────────────────── */}
                <div className="timeline__card">
                  <div className="timeline__card-header">
                    <Icon className="timeline__card-icon" size={20} strokeWidth={1.8} />
                    <h3 className="timeline__card-title">{item.title}</h3>
                  </div>
                  <p className="timeline__card-year">{item.year}</p>
                  <p className="timeline__card-desc">{item.description}</p>
                </div>

                {/* ── Node on the line ─────────────────────────────────── */}
                <div className="timeline__node" aria-hidden="true">
                  <Icon size={18} strokeWidth={2} color="#ffffff" />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Scoped styles ──────────────────────────────────────────────────── */}
      <style jsx>{`
        /* ── Section wrapper ─────────────────────────────── */
        .qualification.section {
          padding: 6rem 0;
          background-color: #000000;
        }

        .container {
          max-width: 1100px;
          margin: 0 auto;
          padding: 0 1.5rem;
        }

        /* ── Section header  (matches portfolio pattern) ─── */
        .section__header {
          text-align: center;
          margin-bottom: 4rem;
        }

        .section__subtitle {
          display: inline-block;
          font-size: 3rem;
          font-weight: 600;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #ca8a04; /* yellow-600 */
          margin-bottom: 0.75rem;
        }

        .section__title {
          font-size: clamp(3rem, 4vw, 3rem);
          font-weight: 700;
          color: #ffffff;
          line-height: 1.25;
        }

        /* ── Timeline container ──────────────────────────── */
        .timeline {
          position: relative;
          max-width: 900px;
          margin: 0 auto;
          padding: 1rem 0 2rem;
        }

        /* Vertical centre line */
        .timeline__line {
          position: absolute;
          left: 50%;
          top: 0;
          bottom: 0;
          width: 2px;
          background: #eab308; /* yellow-500 */
          transform: translateX(-50%);
          border-radius: 2px;
        }

        /* ── Each timeline row ───────────────────────────── */
        .timeline__item {
          position: relative;
          width: 45%;
          margin-bottom: 2.75rem;
        }

        /* Right-side items → float right */
        .timeline__item--right {
          margin-left: 55%;
        }

        /* Left-side items → stay left */
        .timeline__item--left {
          margin-left: 0;
        }

        /* ── Node (yellow circle on the line) ────────────── */
        .timeline__node {
          position: absolute;
          top: 1.6rem;
          width: 46px;
          height: 46px;
          border-radius: 50%;
          background: #eab308;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 0 0 4px #fef9c3; /* yellow-100 halo */
          z-index: 2;
        }

        /* Node position for right-side cards → put it on the LEFT of the card */
        .timeline__item--right .timeline__node {
          left: calc(-55% + -23px + 2px);
          /* centers it on the line: negative half-width of gap + half node */
          left: -55px;
        }

        /* Node position for left-side cards → put it on the RIGHT of the card */
        .timeline__item--left .timeline__node {
          right: -55px;
        }

        /* ── Card ─────────────────────────────────────────── */
        .timeline__card {
          background: #ffffff;
          border: 1.5px solid #e5e7eb; /* gray-200 */
          border-radius: 1rem;
          padding: 1.5rem 1.75rem;
          box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
          transition: box-shadow 0.25s ease, transform 0.25s ease;
        }

        .timeline__card:hover {
          box-shadow: 0 8px 28px rgba(234, 179, 8, 0.18);
          transform: translateY(-3px);
        }

        .timeline__card-header {
          display: flex;
          align-items: center;
          gap: 0.6rem;
          margin-bottom: 0.5rem;
        }

        .timeline__card-icon {
          color: #0d0d0d;
          flex-shrink: 0;
        }

        .timeline__card-title {
          font-size: 1.05rem;
          font-weight: 700;
          color: #0d0d0d;
          margin: 0;
        }

        .timeline__card-year {
          font-size: 0.85rem;
          color: #9ca3af; /* gray-400 */
          margin: 0 0 0.6rem 0;
        }

        .timeline__card-desc {
          font-size: 0.95rem;
          color: #374151; /* gray-700 */
          line-height: 1.65;
          margin: 0;
        }

        /* ── Responsive: stack on mobile ─────────────────── */
        @media (max-width: 700px) {
          .timeline__line {
            left: 22px;
          }

          .timeline__item {
            width: calc(100% - 60px);
            margin-left: 60px !important;
          }

          .timeline__item--right .timeline__node,
          .timeline__item--left .timeline__node {
            left: auto;
            right: auto;
            top: 1.4rem;
            left: -49px;
          }
        }
      `}</style>
    </section>
  );
}
