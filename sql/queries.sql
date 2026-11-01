
-- Query 1: Count orders by status
SELECT
    status,
    COUNT(*) AS total_orders
FROM orders
GROUP BY status;

-- Query 2: Total revenue from completed orders
SELECT
    ROUND(SUM(order_amount), 2) AS total_revenue
FROM orders
WHERE status = 'completed';

-- Query 3: Revenue by city and product category
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Query 4: Revenue by city
SELECT
    c.city,
    COUNT(*) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.city
ORDER BY revenue DESC;


-- Query 5: Revenue by product category
SELECT
    p.category,
    COUNT(*) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY revenue DESC;


-- Query 6: Top 10 customers by spending
SELECT
    o.customer_id,
    c.city,
    COUNT(*) AS total_orders,
    ROUND(SUM(o.order_amount), 2) AS total_spent
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY o.customer_id, c.city
ORDER BY total_spent DESC
LIMIT 10;


-- Query 7: Monthly revenue trend
SELECT
    SUBSTR(order_ts, 1, 7) AS month,
    COUNT(*) AS completed_orders,
    ROUND(SUM(order_amount), 2) AS revenue
FROM orders
WHERE status = 'completed'
GROUP BY SUBSTR(order_ts, 1, 7)
ORDER BY month;
